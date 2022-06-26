// COPYRIGHT (C) 1995-2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

#ifndef __bignum__
#define __bignum__
/*	********************************************
    *					                       *
    *  Abritrary Precision Integer Arithmetic  *
    *					                       *
    ********************************************

Multi-threaded version.
Note (30-Apr-2022): The "multi-thread" update (circa 2016) allows multiple threads
to use indenendent bignums, but not share them. Sharing was assumed to be too costly
in execution time. The update removed certain global static variables, like the
division remainder, and provided them by other means.

Use
---
Bignums are true integers -- not modulo 2 to some power, they grow as large as
necessary to accomodate the results of arithmetic operations. In programs,
bignums are used in much the same way as the built-in integer types, with some
exceptions. The following operators behave as expected:

			+, -, *, /, %, <<, >>, unary -
			==, !=, <, >, <=, >=
			=, +=, -=, *=, /=, <<=, >>=
			<<, >> with streams

Here are the differences between bignums and built-in integers:

	1) Bignum operations cannot overflow (one can in principal run out of
	   memory, but that would be a very big number indeed).
	2) The logical operators and the increment/decrement operators are not
	   implemented.
	3) Bignums cannot directly be used as conditional expressions in
	   statements such as if, while, for, etc., and with the logical operators
	   &&, ||, and !. You must use a comparison operator to produce an int.
	4) Bignums and integer types can freely be mixed in expressions --
	   the integers will automatically be converted to bignums. Bignums cannot
	   automatically be converted to integers, however, so an explicit
	   member function must be used.
	5) The same is true for bignums and floating point types, except that
	   it doesn't work yet due to software build problems. Conversion of
	   floats to bignums discards the fraction.
	6) The right-hand operand of the shift operators must be an integer --
	   it cannot be a Bignum.
	7) The shift count can be negative, and specifies shifting in the
	   other direction.
	8) Shift right is neither an arithmetic shift nor a logical shift --
	   it is an integer divide by a power of 2. This is due to the use of
	   a dSign/magnitude representation rather than 2's complement.
	   
Memory Considerations for the User
----------------------------------
Bignums are allocated on the heap as necessary. The implementation goes to
considerable trouble to insure that these new/delete operatons are kept to a
minimum. Initialization, simple assignment, argument passing, returning values
from functions, negation, and absolute value will never result in allocating a
new Bignum (unless an implicit conversion from a non-zero integer is necessary).
Thus bignums can be assigned, passed by value, returned, etc., as convenient
without any concern for heap consumption and fragmentation.

Consider a simple expression with bignums: a = a + b; In a simple implementation,
one would: 1) allocate a new temporary Bignum to hold the result a + b; 2) delete
the Bignum previously assigned to a; 3) allocate a copy of the temporary result
to assign to a; and 4) delete the temporary result. Thus we have 2 allocs and 2
deletes. The assignment operator has to copy the temporary because it doesn't
know that it is indeed temporary.

In the actual implementation, reference counts are used to avoid making copies,
and the assignment strategy is more sophisticated. The steps are the same as
before, but the step 3 copy just increments a reference count and the step 4
delete just decrements it, resulting in only 1 alloc and 1 delete (assuming that
a pointed to something with a reference count of 1).

In this example with reference counts, a new Bignum is created and the old
one is destroyed each time a = a + b executes. This pattern of use will increase
fragmentation. As a further improvement, if the assignment operator notices that
the variable being assigned has a reference count of 1 and is big enough to hold
the value being assigned, the existing storage is re-used so that no fragmentation
will result from execution of the expression. Of course if the reference count
is > 1, this cannot be done because the storage is being shared.

The final improvement concerns the arithmetic assignment operators. If the
above example is replaced with a += b, and a has a reference count of 1, and a
is large enough to hold the result a + b, then b is added to a directly and no
heap operations are necessary. This behavior is true for all arithmetic assignment
operators.

Speed Considerations for the User
---------------------------------
Speed is reasonable given that everything is written in C and that I was more
interested in keeping the code simple and portable. The only operation that is
very slow is dividing by a number whose magnitude is larger than will fit in
16 bits.

Implementation Overview
-----------------------
A Bignum is a dSign tag and a pointer to a varaible length structure that is
allocated on the heap and encodes the magnitude. Bignums are thus small and can be
passed by value efficiently. The dSign tag can be -1, 0, or +1, meaning that the
integer represented is <0, 0, or >0 respectively. There is some redundancy in the
dSign tag, which will be explained below.

The pointer may be null, meaning that the Bignum is 0. Thus the Bignum value 0
can be represented without any heap storage, a property that is used frequently
to enhance performance. If the pointer is non-null, it points to a structure
consisting of a fixed length header followed by 1 or more words, allocated as one
block on the heap. A "Word" is the atomic unit of data that makes up a Bignum
magnitude. The exact definition of the Word is contained in a typedef that is
set up depending on the properties of the compiler in use � a Word must be an
unsigned integer type that is no larger than half the size in bits of an
unsigned long. Typically, unsigned short or unsigned char would be used.

The header contains a count of the number of words allocated (nwa), the number of
words actually in use (nwu), and a reference count that indicates how many
different bignums are pointing to the structure. The magnitude of the Bignum is
represented by the first nwu words, starting with the least significant. nwu can
be 0, meaning of course that the magnitude is 0. The values of the allocated words
beyond the first nwu are undefined.

When a Bignum is created, enough words must be allocated to hold the magnitude.
Often the exact size needed is not known when the allocation is done, so a
reasonable guess is made. Consider the add operation. The number of words needed
for the result is not known until the add is done, but we need to do the allocation
first, and we don't want to do the add twice. A reasonable guess is to use the
larger of the nwu counts of the two operands, possibly plus one more Word if
an overflow might occur. An overflow might occur if the most significant Word (msw)
of the larger is 0xFFFF, or, if the two nwu's are equal, if the sum of the msw's
of the operands is 0xFFFF. Once the add is complete, we can set nwu of the result
based on the actual most significant non-zero Word.

There is no requirement that the msw of a Bignum be non-zero, although various
routines will operate more efficiently (both space and time) if that is so. As of
this writing, all routines correctly set nwu to the most significant non-zero
Word, but do not assume that the msw is necessarily non-zero. Future routines
may for some reason result in one or more 0 msw's, so all routines must always
assume that possibility.

It can be seen that there are three ways that a Bignum can be 0: the pointer is
null, nwu is 0, or all nwu words are 0. Since it is frequuently necessary to test
for 0, the dSign tag redundantly stores this information. The dSign tag must be
kept consistent with the rest of the Bignum state, or the code will surely break.
I usually hate this kind of redundant information, but here I think efficiency
comes first, and anyway the dSign tag is hidden so problems are localized to
this code. Note that the dSign tag is not in the header so that we can do negatives
and absolute values without allocating a new Bignum. This has turned out to be
very convenient.
*/

#include <limits.h>
#include <iostream>
#include <string>

#define P4_ASM 0
#define KEEP_STATS 0

class Bignum
{   
  // Bignum magnitudes are represented by 0 or more "words". The typedefs are
  // for portability -- Word and DWord must be unsigned and Word  must contain no
  // more than half the number of bits in DWord.
  typedef uint32_t Word;
  typedef uint64_t DWord;
  typedef int64_t SDWord;

  enum
  {
    kWSC = CHAR_BIT * sizeof (Word),	// Word shift count
  };
  signed char dSign;	// 0 means 0, 1 means >0, -1 means <0
  struct BNHeader
  { 
    long
      refc,		// reference count
      nwa,		// number of words allocated
      nwu,		// number of words used
      bpt;		// position of the binary point, in words, + or -
			//   i.e., actual number is 2^(kWSC*bpt) times larger
    Word num[1];	// variable length magnitude
  }
  *dNum;

  // internal functions
  static BNHeader *makebig (long);
  void allocate (long words, int share = 0, bool forceAlloc = false);
  bool allocate2(long words, Bignum& save, bool force = false);
  void constructFromInt (unsigned long);
  void constructFromFloat (long double);
  void constructFromLong (unsigned long long);

  Bignum& addbpt (long);
  // returns	this
  // effect	Add argument to bpt, effectively shifting left by
  //		specififed number of words
  // note	This is fast if this bignum is not shared, and safe
  //		if it is.
  
# ifdef NDEBUG
    void verify (bool = true) const {}
# else
    void verify (bool strict = true) const;
# endif

  static long    bigcmp (const Bignum&, const Bignum&);
  static Bignum& bigadd (const Bignum&, const Bignum&, Bignum&);
  static Bignum& bigsub (const Bignum&, const Bignum&, Bignum&);
  static Bignum& biglsh (const Bignum&, long         , Bignum&);
  static Bignum& bigrsh (const Bignum&, long         , Bignum&);
  static Bignum& bigmul (const Bignum&, const Bignum&, Bignum&);
  static Bignum& bigdiv (const Bignum&, const Bignum&, Bignum&, Bignum*);

  // explicitly callable constructor and destructor
  void destroy ();
  void create (const Bignum&);
  
  // statistics. Not multi-thread correct, but won't break anything.
# if KEEP_STATS
  static unsigned long
    dNZCr,			// number of 0's created
    dNZDs,			// number of 0's destroyed
    dNBCr,			// number of bignums created
    dNBDs,			// number of bignums destroyed
    dNBAl,			// number of bignums allocated
    dNBDl,			// number of bignums deleted
    dLBAl;			// largest Bignum allocated
# endif

public:
  // constructors and destructor
  Bignum (const Bignum& b)	{ this->create (b); };
  Bignum ();

  // Need all these so compiler knows which one to use
  Bignum (int);
  Bignum (long);
  Bignum (unsigned int);
  Bignum (unsigned long);
  Bignum (long long);
  Bignum (unsigned long long);
  Bignum (double);
  Bignum (long double);
  Bignum (const char*);
  Bignum (const std::string&);
 ~Bignum ();
  
  // arithmetic operators
  friend Bignum operator +  (const Bignum&, const Bignum&);
  friend Bignum operator -  (const Bignum&, const Bignum&);
  friend Bignum operator *  (const Bignum&, const Bignum&);
  friend Bignum operator /  (const Bignum&, const Bignum&); // quotient is integer, remainder may include fraction
  friend Bignum operator %  (const Bignum&, const Bignum&);
  friend Bignum operator << (const Bignum&, long);
  friend Bignum operator >> (const Bignum&, long);
  friend Bignum operator -  (Bignum);

  Bignum& mul(const Bignum& a, const Bignum& b) { return bigmul(a, b, *this);}
  Bignum& div(const Bignum& a, const Bignum& b, Bignum& rem) { return bigdiv(a, b, *this, &rem);}
  
  // comparison operators
  friend int operator <  (const Bignum&, const Bignum&);
  friend int operator >  (const Bignum&, const Bignum&);
  friend int operator <= (const Bignum&, const Bignum&);
  friend int operator >= (const Bignum&, const Bignum&);
  friend int operator == (const Bignum&, const Bignum&);
  friend int operator != (const Bignum&, const Bignum&);
  
  // assignment operators
  Bignum& operator =   (const Bignum&);
  Bignum& operator +=  (const Bignum&);
  Bignum& operator -=  (const Bignum&);
  Bignum& operator *=  (const Bignum&);
  Bignum& operator /=  (const Bignum&);
  Bignum& operator <<= (long);
  Bignum& operator >>= (long);
  
  Bignum abs () const;
  // returns    Absolute value of this bignum
  // note       Very short, constant time

  long bits () const;
  // return     The bit position of the MSB, plus 1
  // notes      Equivalently, log base 2 of the magnitude, truncated, plus 1
  //            If this is an integer, bits() returns its size in bits
  //            Returns 0 if this is 0

  long precision () const;
  // returns    The number of bits of precision of the magnitude, i.e. the number
  //            of bits from the least significant non-zero bit to the most
  //            significant non-zero bit, inclusive.

  Bignum integer (bool round = false) const;
  // returns    Integer part, optionally rounded

  int64_t makeint () const;
  // returns    Signed integer equivalent, modulo 2^(# bits in long)

  double makeDouble () const;
  // returns    Floating point equivalent or best estimate
  // note       May overflow. Use frexp() to avoid this.

  long double makeLongDouble () const;
  // returns    Floating point equivalent or best estimate
  // note       May overflow. Use frexp() to avoid this.

  template<typename T>
  T frexp (long& exp) const;
  // returns    x such that 0.5 <= x < 1.0 and x * 2^exp is the best estimate of this
  //            bignum
  // effect     Set exp as defined above

  double log () const;
  // returns    Estimate of log base e

  Bignum power (long n, Bignum mod = 0) const;
  // returns    This raised to the nth power, modulo mod if mod is non-zero
  // notes      Result is 0 if n is negative; for negative powers, use rational numbers
  //            The mod argument can be used for RSA
  //            This does O(log2(n)) multiplies
  
  Bignum gcd (Bignum) const;
  // returns    Greatest common divisor of this and argument
  // notes      Always positive
  //            0 if either value is 0

  static Bignum read (const char*&);
  // returns    Bignum corresponding to a decimal string
  // effect     Update argument to point to terminating character, i.e. first
  //            character beyond end of syntactically legal bignum
  // note       Format: [whitespace][+ or -]<one or more digits>
  
  std::string decimal (bool commas = false) const;
  // returns    Decimal representation of integer part

  std::string primeFact (bool showProgress = false) const;
  // returns	prime factorization suitable for printing

  std::string hex (bool full = false) const;
  // returns    Hex representation of value.
  //		If full, append reference count and num words allocated.
  // notes	Format is sign (+ or -) followed by comma separated list
  //		of hex words. Dot replaces comma if the binary point is
  //		between words. Dot preceeds first word if it's there.
  //		Otherwise, if binary point isn't at right, show it with
  //		"e+-n".

  const char* dump (bool full = false) const;
  // returns    Same as hex, but usable in the debugger's watch window
  // notes      Use "x.dump(0),s" in watch window (argument can also be 1,
  //            but can't be "true" or "false")
  //            Using hex() doesn't work because debugger doesn't display
  //            correctly from newly allocated memory
  //            NOT THREAD-SAFE

  static void print_stats (std::ostream&);
  // effect     Print statistics
};

/*      ************************
        *                      *
        *  Inline Definitions  *
        *                      *
        ************************
*/
inline Bignum& Bignum::operator +=  (const Bignum& a) { return bigadd (*this, a, *this   );}
inline Bignum& Bignum::operator -=  (const Bignum& a) { return bigsub (*this, a, *this   );}
inline Bignum& Bignum::operator *=  (const Bignum& a) { return bigmul (*this, a, *this   );}
inline Bignum& Bignum::operator /=  (const Bignum& a) { return bigdiv (*this, a, *this, 0);}
inline Bignum& Bignum::operator <<= (long a)	      { return biglsh (*this, a, *this   );}
inline Bignum& Bignum::operator >>= (long a)	      { return bigrsh (*this, a, *this   );}

inline int operator == (const Bignum& a, const Bignum& b) { return Bignum::bigcmp (a, b) == 0; }
inline int operator != (const Bignum& a, const Bignum& b) { return Bignum::bigcmp (a, b) != 0; }
inline int operator <  (const Bignum& a, const Bignum& b) { return Bignum::bigcmp (a, b) <  0; }
inline int operator >  (const Bignum& a, const Bignum& b) { return Bignum::bigcmp (a, b) >  0; }
inline int operator <= (const Bignum& a, const Bignum& b) { return Bignum::bigcmp (a, b) <= 0; }
inline int operator >= (const Bignum& a, const Bignum& b) { return Bignum::bigcmp (a, b) >= 0; }

inline Bignum operator - (Bignum a) { a.dSign = -a.dSign; return a; }

inline Bignum Bignum::abs () const { Bignum a = *this; if (a.dSign) a.dSign = 1; return a;}


// I/O operators
std::istream& operator >> (std::istream&, Bignum&);
inline std::ostream& operator << (std::ostream& out, Bignum bn) { return out << bn.decimal().c_str();}

inline Bignum::Bignum ()
{ 
  dSign = 0;
  dNum = 0;
# if KEEP_STATS
      ++dNZCr;
# endif
}

inline Bignum::~Bignum ()
{
  if (dNum)
    destroy ();
# if KEEP_STATS
  else
    ++dNZDs;
# endif
}

inline bool Bignum::allocate2(long words, Bignum& save, bool force)
{
  BNHeader* rn = dNum;
  bool alloc = !rn || rn->nwa < words || rn->refc > 1 || force;
  if (alloc)
  {
    save = *this;
    destroy ();
    dNum = makebig (words);
  }
  return alloc;
}

/*	**********************
	*		     *
	*  Rational Numbers  *
	*		     *
	**********************

Rational is an arithmetic type that has the property the the results of +, -, *,
and / are always exact.
*/

class rational
{   
  Bignum dNum, dDenom;	// numerator and denominator

public:
  // constructors. Note that the (int a) version prevents ambiguity between
  // (long a, long b = 1) and (const char*) when the constant 0 is used.
  rational () {}
  rational (const Bignum& a)                  : dNum(a), dDenom(1) {}
  rational (const Bignum& a, const Bignum& b) : dNum(a), dDenom(b) {}
  rational (long a, long b = 1)               : dNum(a), dDenom(b) {}
  rational (int a)                            : dNum(a), dDenom(1) {}
  rational (const char*);
  rational (const std::string&);
  
  // arithmetic operators
  friend rational operator +  (const rational&, const rational&);
  friend rational operator -  (const rational&, const rational&);
  friend rational operator *  (const rational&, const rational&);
  friend rational operator /  (const rational&, const rational&);
  friend rational operator -  (rational);

  // comparison operators
  friend int operator <  (const rational&, const rational&);
  friend int operator >  (const rational&, const rational&);
  friend int operator <= (const rational&, const rational&);
  friend int operator >= (const rational&, const rational&);
  friend int operator == (const rational&, const rational&);
  friend int operator != (const rational&, const rational&);
  
  // assignment operators
  rational& operator =   (const rational& a) { dNum = a.dNum; dDenom = a.dDenom; return *this; }
  rational& operator +=  (const rational& a) { return *this = *this + a; }
  rational& operator -=  (const rational& a) { return *this = *this - a; }
  rational& operator *=  (const rational& a) { return *this = *this * a; }
  rational& operator /=  (const rational& a) { return *this = *this / a; }
  
  rational& norm ();	// reduce to least common denominator
  
  rational power (long n) const;
  // returns    This raised to the nth power
  // notes      This does O(log2(n)) multiplies
  
  long double makefloat () const;
  // returns    Floating point equivalent or best estimate
  // note       May overflow.

  static rational read (const char*&);
  // returns    rational corresponding to a decimal string
  // effect     Update argument to point to terminating character, i.e. first
  //            character beyond end of syntactically legal bignum
  // note       Format: <bignum>[[whitespace]/<bignum>]
  
  // I/O
  friend std::istream& operator >> (std::istream&, rational&);
  friend std::ostream& operator << (std::ostream&, rational);
  std::string decimal (long precision, bool commas = false) const;
  std::string print() const;  // print as ratio
};


/*    **************************
      *                        *
      *  Performance Counters  *
      *                        *
      **************************
*/

class PerformanceCounter
{
public:
  PerformanceCounter (const char* format);
 ~PerformanceCounter ();

  PerformanceCounter& operator += (double);

private:
  const char*   dFormat;
  unsigned long dCount;
  double        dTotal;
};

class PerformanceProbe
{
public:
  PerformanceProbe (PerformanceCounter&, long&);
 ~PerformanceProbe ();

private:
  PerformanceCounter& dCounter;
  long&               dStatistic;
  long                dStartValue;
};

#endif
