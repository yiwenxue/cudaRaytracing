#pragma once

#include <cuda_runtime.h>

#include "CudaUtils.h"

#include <cassert>

enum class Direction
{
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
};

template <class T>
class CudaArray
{
public:
    CudaArray() = default;
    CudaArray(uint32_t size);
    ~CudaArray();

    void alloc(uint32_t size);
    void release();

    struct BufferUpdate
    {
        uint32_t offset;
        uint32_t size;
    };

    void copy(Direction direction = Direction::HOST_TO_DEVICE, uint32_t offset = 0,
              uint32_t size = 0);

    T *get()
    {
        return m_hptr;
    }

    T *getDevice()
    {
        return m_dptr;
    }

    uint32_t getSize() const
    {
        return m_size;
    }

    uint32_t getByteSize() const
    {
        return m_size * sizeof(T);
    }

private:
    uint32_t m_size{0};
    T       *m_dptr{nullptr};
    T       *m_hptr{nullptr};
};

template <class T>
CudaArray<T>::CudaArray(uint32_t size) : m_size(size)
{
    alloc(size);
}

template <class T>
CudaArray<T>::~CudaArray()
{
    release();
}

template <class T>
void CudaArray<T>::alloc(uint32_t size)
{
    if (m_dptr || m_hptr)
    {
        release();
    }

    m_hptr = static_cast<T *>(malloc(sizeof(T) * size));
    assert(m_hptr && "failed to alloc host memory");
    cudaMalloc((void **) &m_dptr, size * sizeof(T));
    cudaCheckErrors("failed to alloc cuda memory");
    m_size = size;
}

template <class T>
void CudaArray<T>::release()
{
    if (m_dptr || m_hptr)
    {
        free(m_hptr);
        cudaFree(m_dptr);
        cudaCheckErrors("failed to release memory");

        m_dptr = nullptr;
        m_hptr = nullptr;
        m_size = 0;
    }
}

template <class T>
void CudaArray<T>::copy(Direction direction, uint32_t offset, uint32_t size)
{
    if (size == 0)
    {
        size = m_size;
    }

    switch (direction)
    {
        case Direction::HOST_TO_DEVICE:
            cudaMemcpy((void *) (m_dptr + offset), (void *) (m_hptr + offset), size * sizeof(T),
                       cudaMemcpyHostToDevice);
            break;
        case Direction::DEVICE_TO_HOST:
            cudaMemcpy((void *) (m_hptr + offset), (void *) (m_dptr + offset), size * sizeof(T),
                       cudaMemcpyDeviceToHost);
            break;
    }
}
