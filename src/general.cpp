// **************************************
// *                                    *
// *  Various Functions of General Use  *
// *                                    *
// **************************************

#include "general.h"
#include <stdarg.h>
#include <string>

namespace BillGeneral {
  namespace Math
  {
    int64_t gcd(int64_t a, int64_t b)
    {
      a = std::abs(a);
      b = std::abs(b);
      if (a > b)
      {
        a ^= b;
        b ^= a;
        a ^= b;
      }

      if (a == 0)
        return b;

      while (a > 1 && a != b)
      {
        int64_t d = b - a;
        if (d >= a)
          b = d;
        else
        {
          b = a;
          a = d;
        }
      }
      return a;
    }

    int64_t lcm(int64_t a, int64_t b)
    {
      return a / gcd(a, b) * b;
    }
  }

  std::string strFormat(const char* format, ...)
  {
    char buf[4096];
    va_list ap;
    va_start(ap, format);
    vsprintf(buf, format, ap);
    va_end(ap);
    buf[sizeof(buf) - 1] = 0;    // in case the buffer got filled with no room for terminating null
    return buf;
  }

}
