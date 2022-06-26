// **************************************
// *                                    *
// *  Various Functions of General Use  *
// *                                    *
// **************************************

#ifndef _general_
#define _general_

#include <string>

// Bounds check
#define BC(index, size) assert(0 <= (index) && (index) < (size))

namespace BillGeneral {
  namespace Math
  {
    // Integer divide, round to -infinity
    template<class T>
    inline T iDiv(T x, T y)
    {
      static_assert(std::is_integral<T>::value, "iDiv<T> must be integral type");
      return (x >= 0 ? x : x - abs(y) + 1) / y;
    }

    // Integer modulo (result is 0 or its sign matches sign of y).
    template<class T>
    inline T iMod(T x, T y)
    {
      static_assert(std::is_integral<T>::value, "iMod<T> must be integral type");
      return x - iDiv(x, y) * y;
    }

    // Greatest common divisor. a and b can be any integers, result is always non-negative.
    // If one argument is 0, gcd is other argument. 
    int64_t gcd(int64_t a, int64_t b);

    // Least common multiple
    int64_t lcm(int64_t, int64_t);

    std::string printEng(double);
  }

  std::string strFormat(const char* format, ...);

}

#endif
