#ifndef __array__h__
#define __array__h__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tbb/tbb.h"

typedef unsigned int uint;

template<class T>
class array1D
{
public:
	array1D(){data = 0;}
	~array1D() { free(); }
	T* data;
	
	size_t n;
	void alloc(size_t size)
	{
		n = size;
		if(data==0)
			data = (T*) new T [n];
	}
	void free()
	{
		if(data)
			delete [] data;
		data = 0;
	}
	array1D<T>& operator*=(const T &c)
	{
		tbb::parallel_for((int)0,(int)n,(int)1,[&](int i)
		{
			this->data[i] *= c;
		});
		return *this;
	}
    array1D<T>& operator/=(const T &c)
    {
        tbb::parallel_for((int)0,(int)n,(int)1,[&](int i)
        {
            this->data[i] /= c;
        });
        return *this;
    }
	array1D<T>& operator+=(const array1D<T> &x)
	{
		tbb::parallel_for((int)0,(int)x.n,(int)1,[&](int i)
		{
			this->data[i] += x.data[i];
		});
		return *this;
	}
	array1D<T>& operator-=(const array1D<T> &x)
	{
		tbb::parallel_for((int)0,(int)x.n,(int)1,[&](int i)
		{
			this->data[i] -= x.data[i];
		});
		return *this;
	}
	size_t getSize(){return n;}
	size_t getTypeSize() {return sizeof(T); }
	T* getPtr() const {return data;}
	void setZero() { memset(data, 0, sizeof(T)*n); }
};
typedef array1D<float> arrayf;
typedef array1D<double> arrayd;
typedef array1D<char> arrayc;
#endif