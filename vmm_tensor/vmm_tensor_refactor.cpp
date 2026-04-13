#include <cuda.h>
#include <torch/extension.h>

#include <atomic>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    CUresult err = call;                                                   \
    if (err != CUDA_SUCCESS) {                                             \
      const char* err_str = "";                                            \
      cuGetErrorString(err, &err_str);                                     \
      throw std::runtime_error(std::string("CUDA error: ") + err_str);     \
    }                                                                      \
  } while (0)

namespace py = pybind11;

class VMMTensorImpl {
 public:
  VMMTensorImpl(int64_t reserve_size_align_2mb, torch::Dtype dtype,
                int64_t device_id, int64_t num_attn_layers)
      : size_(reserve_size_align_2mb),
        dtype_(dtype),
        device_id_(device_id),
        num_attn_layers_(num_attn_layers) {
    CHECK_CUDA(cuInit(0));
    compute_size();
    get_alignment();
    for (int i = 0; i < num_attn_layers_; ++i) {
      layer_reserve();
    }
  }

  ~VMMTensorImpl() {
    for (auto& vaddr : vaddrs_) {
      if (vaddr) {
        cuMemUnmap(vaddr, 2 * size_bytes_);
        cuMemAddressFree(vaddr, 2 * size_bytes_);
      }
    }
    while (!allocated_handles_.empty()) {
      cuMemRelease(allocated_handles_.back());
      allocated_handles_.pop_back();
    }
  }

  std::vector<torch::Tensor> create_tensors(
      int64_t alloc_align_size, const std::vector<int64_t>& tensor_shape) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(vaddrs_.size());
    for (size_t i = 0; i < vaddrs_.size(); ++i) {
      allocate_map_set_access(0, alloc_align_size, i);
      allocate_map_set_access(size_bytes_, alloc_align_size, i);
      tensors.push_back(tensor(0, tensor_shape, i));
    }
    return tensors;
  }

  void batch_allocate_async(
      const std::vector<std::pair<size_t, int64_t>>& ops) {
    batch_memory_op_running_ = true;
    std::thread([this, ops]() {
      int before_size = allocated_handles_.size();
      int handles_per_op = static_cast<int>(vaddrs_.size()) * 2;
      CUmemAllocationProp prop = default_prop();
      for (const auto& [offset, alloc_align_size] : ops) {
        size_t new_size_bytes = compute_size_for_shape(alloc_align_size);
        for (int i = 0; i < handles_per_op; ++i) {
          CUmemGenericAllocationHandle handle;
          CHECK_CUDA(cuMemCreate(&handle, new_size_bytes, &prop, 0));
          allocated_handles_.push_back(handle);
        }
      }

      int op_index = 0;
      for (const auto& [offset, alloc_align_size] : ops) {
        size_t new_size_bytes = compute_size_for_shape(alloc_align_size);
        for (int i = 0; i < handles_per_op; ++i) {
          size_t actual_offset = (i % 2) ? offset + size_bytes_ : offset;
          CHECK_CUDA(cuMemMap(
              vaddrs_[i / 2] + actual_offset, new_size_bytes, 0,
              allocated_handles_[before_size + op_index * handles_per_op + i],
              0));
          CUmemAccessDesc access_desc = {};
          access_desc.location = default_prop().location;
          access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
          CHECK_CUDA(cuMemSetAccess(vaddrs_[i / 2] + actual_offset,
                                    new_size_bytes, &access_desc, 1));
        }
        ++op_index;
      }
      batch_memory_op_running_ = false;
    }).detach();
  }

  void batch_free_async(const std::vector<std::pair<size_t, int64_t>>& ops) {
    batch_memory_op_running_ = true;
    std::thread([this, ops]() {
      int handles_per_op = static_cast<int>(vaddrs_.size()) * 2;
      for (auto op_it = ops.rbegin(); op_it != ops.rend(); ++op_it) {
        const auto& [offset, free_align_size] = *op_it;
        for (int i = handles_per_op - 1; i >= 0; --i) {
          size_t actual_offset = (i % 2) ? offset + size_bytes_ : offset;
          unmap_release(actual_offset, free_align_size, i / 2);
        }
      }
      batch_memory_op_running_ = false;
    }).detach();
  }

  bool is_batch_memory_op_running() const { return batch_memory_op_running_; }

 private:
  void layer_reserve() {
    CUdeviceptr vaddr;
    CHECK_CUDA(cuMemAddressReserve(&vaddr, 2 * size_bytes_, alignment_, 0, 0));
    vaddrs_.push_back(vaddr);
  }

  void allocate_map_set_access(size_t offset, int64_t alloc_align_size,
                               size_t reserve_index) {
    size_t new_size_bytes = compute_size_for_shape(alloc_align_size);
    CUmemAllocationProp prop = default_prop();
    CUmemGenericAllocationHandle handle;
    CHECK_CUDA(cuMemCreate(&handle, new_size_bytes, &prop, 0));
    allocated_handles_.push_back(handle);
    CHECK_CUDA(
        cuMemMap(vaddrs_[reserve_index] + offset, new_size_bytes, 0, handle, 0));
    CUmemAccessDesc access_desc = {};
    access_desc.location = default_prop().location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA(cuMemSetAccess(vaddrs_[reserve_index] + offset, new_size_bytes,
                              &access_desc, 1));
  }

  void unmap_release(size_t offset, int64_t free_align_size,
                     size_t reserve_index) {
    size_t free_size_bytes = compute_size_for_shape(free_align_size);
    CHECK_CUDA(cuMemUnmap(vaddrs_[reserve_index] + offset, free_size_bytes));
    CUmemGenericAllocationHandle handle = allocated_handles_.back();
    allocated_handles_.pop_back();
    CHECK_CUDA(cuMemRelease(handle));
  }

  torch::Tensor tensor(size_t offset, const std::vector<int64_t>& shape,
                       size_t reserve_index) {
    auto options = torch::TensorOptions().dtype(dtype_).device(
        torch::kCUDA, device_id_);
    auto deleter = [](void* ptr) {};
    return torch::from_blob(
        reinterpret_cast<void*>(vaddrs_[reserve_index] + offset), shape,
        deleter, options);
  }

  size_t compute_size_for_shape(int64_t alloc_align_size) const {
    return alloc_align_size * 2 * 1024 * 1024;
  }

  CUmemAllocationProp default_prop() const {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id_;
    return prop;
  }

  void get_alignment() {
    CUmemAllocationProp prop = default_prop();
    CHECK_CUDA(cuMemGetAllocationGranularity(
        &alignment_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  }

  void compute_size() { size_bytes_ = size_ * 2 * 1024 * 1024; }

  int64_t size_;
  torch::Dtype dtype_;
  int64_t device_id_;
  int64_t num_attn_layers_;
  size_t size_bytes_{0};
  size_t alignment_{0};
  std::vector<CUdeviceptr> vaddrs_;
  std::vector<CUmemGenericAllocationHandle> allocated_handles_;
  std::atomic<bool> batch_memory_op_running_{false};
};

PYBIND11_MODULE(_C, m) {
  py::class_<VMMTensorImpl>(m, "VMMTensor")
      .def(py::init<int64_t, torch::Dtype, int64_t, int64_t>())
      .def("create_tensors", &VMMTensorImpl::create_tensors,
           py::call_guard<py::gil_scoped_release>())
      .def("batch_allocate_async", &VMMTensorImpl::batch_allocate_async,
           py::call_guard<py::gil_scoped_release>())
      .def("batch_free_async", &VMMTensorImpl::batch_free_async,
           py::call_guard<py::gil_scoped_release>())
      .def("is_batch_memory_op_running",
           &VMMTensorImpl::is_batch_memory_op_running);
}
