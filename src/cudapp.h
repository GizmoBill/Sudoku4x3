#ifndef _cudapp_
#define _cudapp_

// *******************************************
// *                                         *
// *  Safer C++ Wrapping of Some Basic Cuda  *
// *                                         *
// *******************************************

#include <string>
#include <stdexcept>

// *************
// *           *
// *  Streams  *
// *           *
// *************

class CudaStream
{
public:
  CudaStream() { cudaStreamCreate(&stream_); }

  ~CudaStream() { cudaStreamDestroy(stream_); }

  cudaStream_t& stream() { return stream_; }

private:
  cudaStream_t stream_;
};

// *********************
// *                   *
// *  Safe Primitives  *
// *                   *
// *********************

inline void checkErr(const cudaError_t& err)
{
  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(err));
}

template<typename T>
void safeAlloc(T*& array, size_t size)
{
  checkErr(cudaMalloc(&array, size * sizeof(T)));
}

template<typename T>
void safeManagedAlloc(T*& array, size_t size)
{
  checkErr(cudaMallocManaged(&array, size * sizeof(T)));
}

void safeFree(void* array)
{
  checkErr(cudaFree(array));
}

inline void safeSync()
{
  checkErr(cudaDeviceSynchronize());
}

inline void checkLaunch()
{
  checkErr(cudaGetLastError());
}

// ***********************************
// *                                 *
// *  Device or Uniform Memory Base  *
// *                                 *
// ***********************************

template<typename T>
class CudaMemory
{
public:
  virtual ~CudaMemory() { if (mem_) safeFree(mem_); }

  void alloc(size_t numElements)
  {
    if (num_ == numElements)
      return;

    if (num_ > 0)
      safeFree(mem_);

    num_ = numElements;
    alloc_(mem_, numElements * sizeof(T));
  }

  size_t numElements() const { return num_; }

  size_t numBytes() const { return num_ * sizeof(T); }

  T& operator[](size_t index) { return mem_[index]; }
  const T& operator[](size_t index) const { return mem_[index]; }

  T* mem() { return mem_; }
  const T* mem() const { return mem_; }

  void copyTo(const void* src)
  {
    cudaMemcpy(this->mem_, src, this->numBytes(), cudaMemcpyHostToDevice);
  }

  void prefetchAsync()
  {
    int deviceId = cudaGetDevice(&deviceId);
    checkErr(cudaMemPrefetchAsync(mem_, numBytes(), deviceId));
  }

protected:
  CudaMemory() : num_(0), mem_(nullptr) {}

  CudaMemory(size_t numElements) : num_(numElements), mem_(nullptr) {}

  CudaMemory(const CudaMemory&) = delete;
  CudaMemory& operator=(const CudaMemory&) = delete;

  CudaMemory& operator=(const CudaMemory&& cm)
  {
    if (mem_)
      safeFree(mem_);
    mem_ = cm.mem_;
    num_ = cm.num_;
    cm.mem_ = 0;
    cm.num_ = 0;
  }

  CudaMemory(const CudaMemory&& cm)
  {
    *this = cm;
  }

  virtual void alloc_(T*& mem, size_t numElements) = 0;

  size_t num_;
  T* mem_;
};

// *******************
// *                 *
// *  Device Memory  *
// *                 *
// *******************

template<typename T>
class CudaDeviceMemory : public CudaMemory<T>
{
public:
  CudaDeviceMemory() = default;

  CudaDeviceMemory(size_t numElements)
    : CudaMemory<T>(numElements)
  {
    alloc_(this->mem_, numElements);
  }

protected:
  virtual void alloc_(T*& mem, size_t numElements) override
  {
    safeAlloc(mem, numElements);
  }
};

// ********************
// *                  *
// *  Unified Memory  *
// *                  *
// ********************

template<typename T>
class CudaUnifiedMemory : public CudaMemory<T>
{
public:
  CudaUnifiedMemory() = default;

  CudaUnifiedMemory(size_t numElements)
    : CudaMemory<T>(numElements)
  {
    alloc_(this->mem_, numElements);
  }

protected:
  virtual void alloc_(T*& mem, size_t numElements) override
  {
    safeManagedAlloc(mem, numElements);
  }
};

#endif
