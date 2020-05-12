#pragma once

#include <type_traits>
#include <cmath>
#include <cstdint>
#include "vector.cuh"

#define DEVICE __forceinline__ __device__

template <typename Value_, size_t Size_> struct Matrix {
    using Value = Value_;
    using Column = Array<Value, Size_>;
    static constexpr size_t Size = Size_;

    DEVICE Matrix() { }

    Matrix(const Matrix &) = default;

    template <typename T>
    DEVICE Matrix(const Matrix<T, Size> &a) {
        for (size_t i = 0; i < Size; ++i)
            m[i] = (Column) a.m[i];
    }

    DEVICE Matrix operator-() const {
        Matrix result;
        for (size_t i = 0; i < Size; ++i)
            result[i] = -m[i];
        return result;
    }

    DEVICE friend Matrix operator+(const Matrix &a, const Matrix &b) {
        Matrix result;
        for (size_t i = 0; i < Size; ++i)
            result[i] = a.m[i] + b.m[i];
        return result;
    }

    DEVICE Matrix& operator+=(const Matrix &a) {
        for (size_t i = 0; i < Size; ++i)
            m[i] += a.m[i];
        return *this;
    }

    DEVICE friend Matrix operator-(const Matrix &a, const Matrix &b) {
        Matrix result;
        for (size_t i = 0; i < Size; ++i)
            result[i] = a.m[i] - b.m[i];
        return result;
    }

    DEVICE Matrix& operator-=(const Matrix &a) {
        for (size_t i = 0; i < Size; ++i)
            m[i] -= a.m[i];
        return *this;
    }

    DEVICE friend Matrix operator*(const Matrix &a, const Matrix &b) {
        Matrix result;
        for (size_t j = 0; j < Size; ++j) {
            Column sum = a[0] * b[0][j];
            for (size_t i = 1; i < Size; ++i)
                sum += b[i][j] * a[i];
            result[j] = sum;
        }
    }

    DEVICE bool operator==(const Matrix &a) const {
        for (size_t i = 0; i < Size; ++i) {
            if (m[i] != a.m[i])
                return false;
        }
        return true;
    }

    DEVICE bool operator!=(const Matrix &a) const {
        return !operator==(a);
    }

    DEVICE const Column &operator[](size_t i) const {
        assert(i < Size);
        return m[i];
    }

    DEVICE Column &operator[](size_t i) {
        assert(i < Size);
        return m[i];
    }

    Column m[Size];
};

// Import some common Enoki types
using Matrix3f = Matrix<float, 3>;
using Matrix4f = Matrix<float, 4>;

struct Transform4f {
    Matrix4f matrix;
    Matrix4f inverse_transpose;

    DEVICE Transform4f(const float data[32]) {
        for (size_t i = 0; i < 4; i++)
            for (size_t j = 0; j < 4; j++)
                matrix[i][j] = data[i * 4 + j];
        for (size_t i = 0; i < 4; i++)
            for (size_t j = 0; j < 4; j++)
                inverse_transpose[i][j] = data[16 + i * 4 + j];
    }
};

DEVICE Vector3f transform_point(const Transform4f &t, const Vector3f &p) {
    Vector4f result = t.matrix[3];
    for (size_t i = 0; i < 3; ++i)
        result += p[i] * t.matrix[i];
    return Vector3f(result.x(), result.y(), result.z());
}

DEVICE Vector3f transform_normal(const Transform4f &t, const Vector3f &n) {
    Vector4f result = t.inverse_transpose[0];
    result *= n.x();
    for (size_t i = 1; i < 3; ++i)
        result += n[i] * t.inverse_transpose[i];
    return Vector3f(result.x(), result.y(), result.z());
}

DEVICE Vector3f transform_vector(const Transform4f &t, const Vector3f &v) {
    Vector4f result = t.matrix[0];
    result *= v.x();
    for (size_t i = 1; i < 3; ++i)
        result += v[i] * t.matrix[i];
    return Vector3f(result.x(), result.y(), result.z());
}