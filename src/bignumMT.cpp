// COPYRIGHT (C) 1995-2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

/*      ********************************************
        *                                          *
        *  Abritrary Precision Integer Arithmetic  *
        *                                          *
        ********************************************
*/
#include <bignumMT.h>
#include <general.h>
#include "float.h"

#include <vector>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace BillGeneral;

PerformanceCounter zeros ("%d calls, %.0f zeros created\n");

/*      ************************************
        *                                  *
        *  Verify Representation is Legal  *
        *                                  *
        ************************************
*/
#ifndef NDEBUG
void Bignum::verify (bool strict) const
{
  if (dNum)
  {
    assert (dNum->refc > 0);
    assert (0 <= dNum->nwu && dNum->nwu <= dNum->nwa);

    if (dSign)
    {
      assert (dNum->nwu > 0);

      if (strict)
      {
        assert (dNum->num[0]);
        assert (dNum->num[dNum->nwu - 1]);
      }
      else
      {
        long i;
        for (i = dNum->nwu - 1; i >= 0 && !dNum->num[i]; --i);
        assert (i >= 0);
      }
    }
    else
    {
      if (strict)
        assert (!dNum->nwu);
      else
        for (int i = 0; i < dNum->nwu; ++i)
          assert (!dNum->num[i]);
    }
  }
  else
    assert (!dSign);
}
#endif

/*      ****************
        *              *
        *  Statistics  *
        *              *
        ****************
*/
#if KEEP_STATS
unsigned long
  Bignum::dNZCr = 0,		// number of 0's created
  Bignum::dNZDs = 0,		// number of 0's destroyed
  Bignum::dNBCr = 0,		// number of bignums created
  Bignum::dNBDs = 0,		// number of bignums destroyed
  Bignum::dNBAl = 0,		// number of bignums allocated
  Bignum::dNBDl = 0,		// number of bignums deleted
  Bignum::dLBAl = 0;		// largest Bignum allocated
#endif

/*      *************************
        *                       *
        *  Allocation Routines  *
        *                       *
        *************************
*/

Bignum::BNHeader *Bignum::makebig (long n)
{
  assert (n >= 0);
  if (!n) return NULL;

  n += 8;
  // get n words in addition to the one already present in BNHeader, to guarantee one
  // extra hidden word for buffer overreads in ASM code
  BNHeader *bnh =
    (BNHeader *) new char[sizeof (BNHeader) + n * sizeof (Word)]; // not (n-1) to 
  bnh->nwa = n;
  bnh->refc = 1;
  bnh->nwu = 0;
  bnh->bpt = 0;

# if KEEP_STATS
  ++dNBAl, ++dNBCr;
  if ((unsigned long)n > dLBAl)
    dLBAl = n;
# endif

  return bnh;
}

void Bignum::allocate (long words, int share, bool forceAlloc)
{
  if (!dNum)
  {
    dNum = makebig (words);
#   if KEEP_STATS
    if (dNum)
      ++dNZDs;
#   endif
  }
  else
  {
    forceAlloc = forceAlloc || dNum->refc > share + 1;
    if (dNum->nwa < words || /*dNum->nwa > 2 * words ||*/ forceAlloc)
    {
      destroy ();
      dNum = makebig (words);
#     if KEEP_STATS
      if (!dNum)
        ++dNZCr;
#     endif
    }
  }

  if (!dNum)
    dSign = 0;
}


/*      *********************************
        *                               *
        *  Constructors and Destructor  *
        *                               *
        *********************************
*/
void Bignum::constructFromInt (unsigned long n)
{ 
  if (n)
  { 
    dNum = makebig ((sizeof (n) + sizeof (Word) - 1) / sizeof (Word));
    Word *p = dNum->num;
    while (!(Word)n)
      n = (Word)((DWord)n >> kWSC), ++dNum->bpt;
    while (n)
    {
      *p++ = (Word) n;
      ++dNum->nwu;
      n = (Word)((DWord)n >> kWSC);
    }
  }
  else
  {
    dSign = 0;
    dNum = 0;
#   if KEEP_STATS
    ++dNZCr;
#   endif
  }
  verify();
}

void Bignum::constructFromLong (unsigned long long n)
{ 
  if (n)
  { 
    dNum = makebig ((sizeof (n) + sizeof (Word) - 1) / sizeof (Word));
    Word *p = dNum->num;
    while (!(Word)n)
      n >>= kWSC, ++dNum->bpt;
    while (n)
    {
      *p++ = (Word) n;
      ++dNum->nwu;
      n >>= kWSC;
    }
  }
  else
  {
    dSign = 0;
    dNum = 0;
#   if KEEP_STATS
    ++dNZCr;
#   endif
  }
  verify();
}

void Bignum::constructFromFloat (long double x)
{ 
  if (x == 0.)
  { 
    dSign = 0; 
    dNum = 0;
#   if KEEP_STATS
    ++dNZCr;
#   endif
    return;
  }

  if (x < 0.)
    dSign = -1, x = -x;
  else
    dSign = 1;

  int e;
  std::frexp (x, &e);
  dNum = makebig ((sizeof (long double) + sizeof (Word) - 1) / sizeof (Word) + 1);
  dNum->bpt = Math::iDiv (e - 1, (int)kWSC) - dNum->nwa + 1;
  x = std::ldexp (x, -dNum->bpt * kWSC);

  long double y = ldexpl(1., kWSC);
  Word *p = dNum->num;
  for (long i = 0; i < dNum->nwa; ++i, x = ldexpl (x, -kWSC))
  {
    *p = (Word) std::fmod (x, y);
    if (*p || p != dNum->num)
      ++p;
    else
      ++dNum->bpt;
  }

  dNum->nwu = (long)(p - dNum->num);
  verify();
}

Bignum::Bignum (int n)
{
  if (n < 0)
    dSign = -1, n = -n;
  else
    dSign = 1;
  constructFromInt (n);
}

Bignum::Bignum (long n)
{
  if (n < 0)
    dSign = -1, n = -n;
  else
    dSign = 1;
  constructFromInt (n);
}

Bignum::Bignum (long long n)
{
  if (n < 0)
    dSign = -1, n = -n;
  else
    dSign = 1;
  constructFromLong (n);
}

Bignum::Bignum (unsigned int n)
{
  dSign = 1;
  constructFromInt (n);
}

Bignum::Bignum (unsigned long n)
{
  dSign = 1;
  constructFromInt (n);
}

Bignum::Bignum (unsigned long long n)
{
  dSign = 1;
  constructFromLong (n);
}

Bignum::Bignum (double x)
{
  constructFromFloat (x);
}

Bignum::Bignum (long double x)
{
  constructFromFloat (x);
}

Bignum::Bignum (const char* p)
{
  constructFromInt(0);
  *this = read(p);
}

Bignum::Bignum (const std::string& s)
{
  constructFromInt(0);
  const char* p = s.c_str();
  *this = read(p);
}


// Constructors for initialization, argument passing
void Bignum::create (const Bignum &b)
{
  dSign = b.dSign;
  if (dNum = b.dNum)
  {
    ++dNum->refc;
#   if KEEP_STATS
    ++dNBCr;
#   endif
  }
# if KEEP_STATS
  else
    ++dNZCr;
# endif
}

// Destructors
void Bignum::destroy ()
{
  if (dNum)
  {
#   if KEEP_STATS
    ++dNBDs;
#   endif
    if (!--dNum->refc)
    {
      delete [] (char*)dNum;
#     if KEEP_STATS
      ++dNBDl;
#     endif
    }
  }
# if KEEP_STATS
  else
    ++dNZDs;
# endif
}

/*      ************************************
        *                                  *
        *  Conversion to Arithmetic Types  *
        *                                  *
        ************************************
*/
int64_t Bignum::makeint () const
{ 
  if (!dSign)
    return 0;

  int64_t n = 0;
  int i = (sizeof(int64_t) + sizeof(Word) - 1) / sizeof(Word);
  if (i > dNum->nwu)
    i = dNum->nwu;
  Word *p = dNum->num + i;
  while (i--)
    n = (n << kWSC) + *--p;

  if (dSign < 0)
    n = -n;

  for (i = dNum->bpt; i > 0 && n; --i)
    n <<= kWSC;
  for (; i < 0 && n && n != -1; ++i)
    n >>= kWSC;

  return n;
}

template<typename T>
T Bignum::frexp(long& exp) const
{
  if (!dSign)
  {
    exp = 0;
    return 0.;
  }

  long i = dNum->nwu;
  const Word* p = dNum->num + i;
  while (i && !p[-1])
    --p, --i;
  const int n = (sizeof(T) + sizeof(Word) - 1) / sizeof(Word) + 1;
  if (i > n)
    i = n;

  T x = 0.;
  while (i--)
    x = ldexp (x, kWSC) + *--p;

  int e;
  x = ::frexp(x, &e);
  exp = e + (long)(p - dNum->num + dNum->bpt) * kWSC;
  return x;
}


double Bignum::makeDouble () const
{
  long e;
  double x = frexp<double>(e);
  return ldexp(x, e);
}

long double Bignum::makeLongDouble () const
{
  long e;
  long double x = frexp<long double>(e);
  return ldexpl(x, e);
}

double Bignum::log() const
{
  if (!dSign)
    return 0.;

  long e;
  double x = frexp<double>(e);
  return ::log(x) + ::log(2.) * e;
}

/*      **************************
        *                        *
        *  Assignment Operators  *
        *                        *
        **************************
*/
Bignum& Bignum::operator = (const Bignum& a)
{ 
  if (this != &a)
    if (dNum && dNum->refc == 1 && (!a.dNum || dNum->nwa >= a.dNum->nwu))
      if (a.dNum)
      { 
        Word *pa = a.dNum->num, *pt = dNum->num;
        for (long i = a.dNum->nwu; i; --i)
          *pt++ = *pa++;
        dNum->nwu = a.dNum->nwu;
        dNum->bpt = a.dNum->bpt;
        dSign = a.dSign;
      }
      else
        dNum->nwu = 0, dSign = 0, dNum->bpt = 0;
    else
      this->destroy (), this->create (a);

  verify();
  return *this;
}

/*      *********
        *       *
        *  Add  *
        *       *
        *********

This utility routine does the work for + and +=. The result may be a or b
or both or neither.
*/

Bignum& Bignum::bigadd (const Bignum& a, const Bignum& b, Bignum& r)
{
  // If one of the operands is 0, return the other. If the signs of the operands
  // are different, do a subtract
  if (!a.dSign)
    return r = b;
  if (!b.dSign)
    return r = a;
  if (a.dSign != b.dSign)
    return bigsub (a, -b, r);
  
  // Set lg to the largest and sm to the smallest of the operands based on number
  // of words used and the binary point. Note that this is not neseccarily the
  // largest/smallest magnitude, but that's not necessary.
  BNHeader *lg = a.dNum, *sm = b.dNum;
  if (lg->nwu + lg->bpt < sm->nwu + sm->bpt)
    lg = b.dNum, sm = a.dNum;

  long
    bpt = std::min (lg->bpt, sm->bpt),
    lgu = lg->nwu,
    lgz = lg->bpt - bpt,
    smu = sm->nwu,
    smz = sm->bpt - bpt;
    
  // For the result we need the number of words used by lg, plus 1 if
  // a carry might be generated based on looking at the high order words. If the
  // result will fit in r, use it; otherwise make a new one.
  long nwu = lgu + lgz + (long)(((DWord)lg->num[lgu-1] + 
                                 (DWord)(lgu + lgz == smu + smz ? sm->num[smu-1] : 0) +
                                 1) >> kWSC);
  Word
    *plg = lg->num,
    *psm = sm->num;

  // can destroy r even if r is a or b or both because we have saver
  Bignum saver;
  r.allocate2(nwu, saver, r.dNum == lg && lgz || r.dNum == sm && smz);
  Word* pr  = r .dNum->num;

  // Now we can add the operands

  // Loops to handle least significant words of operands with differing binary points
  // Note that no more than one of smz and lgz will be non-zero
  lgu -= smz;
  if (pr == plg)	  
  {
     pr  += smz;
     plg += smz;
  }
  else
    while (smz--)
      *pr++ = *plg++;

  if (pr == psm)
  {
    long skip = std::min (lgz, smu);
    pr  += skip;
    psm += skip;
    smu -= skip;
    lgz -= skip;
  }
  else
    for (; lgz && smu; --lgz, --smu)
      *pr++ = *psm++;

  r.dNum->bpt = bpt;

  // If there are some large 0's left over, it means all of the small words are to
  // the right of all of the large words
  for (; lgz; --lgz)
    *pr++ = 0;

# if !P4_ASM
  // Add both operands, moving the binary point instead of writing least significant 0's
  DWord sum = 0;
  for (; pr == r.dNum->num && smu; --smu, --lgu)
  {
    sum += (DWord)*plg++ + *psm++;
    if (*pr = (Word) sum)
      ++pr;
    else
      ++r.dNum->bpt;
    sum >>= kWSC;
  }

  // Add both operands
  for (; smu; --smu, --lgu)
  {
    sum += (DWord)*plg++ + *psm++;
    *pr++ = (Word) sum;
    sum >>= kWSC;
  }

  // Propagate carry to larger operand, moving the binary point instead of writing least
  // significant 0's
  for (; pr == r.dNum->num && lgu; --lgu)
  { 
    sum += *plg++;
    if (*pr = (Word) sum)
      ++pr;
    else
      ++r.dNum->bpt;
    sum >>= kWSC;
  }

  // Propagate carry to larger operand
  for (; lgu; --lgu)
  { 
    sum += *plg++;
    *pr++ = (Word) sum;
    sum >>= kWSC;
  }
  if (sum)
    *pr++ = 1;

# else
  bpt = 0;
  int moveBpt = pr == r.dNum->num;
  __asm
  {
    // eax	  sum
    // ecx	  loop counter
    // edx	  psm
    // esi	  plg
    // edi	  pr

    mov edx, psm
    mov esi, plg
    mov edi, pr
    mov ecx, smu
    lea edx, [edx+ecx*4]
    lea esi, [esi+ecx*4]
    lea edi, [edi+ecx*4]
    neg ecx
    mov eax, moveBpt
    jnz someSmu
    test eax, 1		  // also clears CF
    jz start4
    jmp start2
someSmu:
    test eax, 1
    jz loop3

    // Add both operands, moving the binary point instead of writing
    // least significant 0's
loop1:
    mov eax, [edx+ecx*4]
    adc eax, [esi+ecx*4]
    jnz enter3
    inc bpt		  // inc and lea don't affect CF
    lea edi, [edi-4]
    inc ecx
    jnz loop1

    // Propagate carry to larger operand, moving the binary point instead
    // of writing least significant 0's
start2:
    rcl al, 1	    // save carry flag
    mov ecx, lgu
    sub ecx, smu
    jz fix
    lea esi, [esi+ecx*4]
    lea edi, [edi+ecx*4]
    neg ecx
    rcr al, 1	    // restore carry flag
loop2:
    mov eax, [esi+ecx*4]
    adc eax, 0
    jnz enter4
    inc bpt
    lea edi, [edi-4]
    inc ecx
    jnz loop2

    jmp last
fix:
    rcr al, 1	    // restore carry flag
    jmp last

    // Add both operands
loop3:
    mov eax, [edx+ecx*4]
    adc eax, [esi+ecx*4]
enter3:
    mov [edi+ecx*4], eax
    inc ecx
    jnz loop3

    // Propagate carry to larger operand
start4:
    rcl al, 1	    // save carry flag
    mov ecx, lgu
    sub ecx, smu
    jz fix
    lea esi, [esi+ecx*4]
    lea edi, [edi+ecx*4]
    neg ecx
    rcr al, 1	    // restore carry flag
loop4:
    mov eax, [esi+ecx*4]
    adc eax, 0
enter4:
    mov [edi+ecx*4], eax
    inc ecx
    jnz loop4
    
last:
    jnc done
    mov dword ptr[edi], 1
    lea edi, [edi+4]
done:
    mov pr, edi
  }
  r.dNum->bpt += bpt;
# endif

  r.dNum->nwu = (long)(pr - r.dNum->num);
  r.dSign = a.dSign;
  r.verify();
  return r;
}

Bignum operator + (const Bignum& a, const Bignum& b)
{ Bignum r;
  return Bignum::bigadd (a, b, r);
}

/*      **************
        *            *
        *  Subtract  *
        *            *
        **************
*/
Bignum& Bignum::bigsub (const Bignum& a, const Bignum& b, Bignum& r)
{ 
  // If one of the operands is 0, return the other. If the signs of the operands
  // are different, do an add
  if (!a.dSign)
    return r = -b;
  if (!b.dSign)
    return r = a;
  if (a.dSign != b.dSign)
    return r = a + -b;
  
  // We need to determine which operand has the larger magnitude, and how many
  // words will be needed in the result. Note that determining the latter may
  // save some memory but is not strictly necessary, but it's essentially free
  // if we're going to determine the larger magnitude anyway. This is all done
  // by bigcmp
  long ru = Bignum::bigcmp (a, b);

  // If the operands are ==, return 0. Otherwise, set up to subtract the smaller
  // magnitude from the larger, and figure out the sign of the result
  if (!ru)
    return r = Bignum();
  Bignum lg, sm;
  if (ru < 0)
    lg = b, sm = a, r.dSign = -a.dSign, ru = -ru;
  else
    lg = a, sm = b, r.dSign =  a.dSign;

  long bpt = std::min (a.dNum->bpt, b.dNum->bpt);
  ru -= bpt;
  long
    lgz = lg.dNum->bpt - bpt,
    smz = sm.dNum->bpt - bpt,
    smu = sm.dNum->nwu;

  // can destroy r even if r is a or b or both because we have the values lg and sm
  r.allocate (ru, (&r == &a) + (&r == &b),
              r.dNum == lg.dNum && lgz || r.dNum == sm.dNum && smz);

  // Now we can subtract the operands
  Word
    *plg = lg.dNum->num,
    *psm = sm.dNum->num,
    *pr  = r .dNum->num;
  SDWord diff = 0;

  // Loops to handle least significant words of operands with differing binary points
  long i = lgz + smz;
  if (pr == plg)	  // no more than one of smz and lgz will be non-zero
  {
    pr  += smz;
    plg += smz;
  }
  else
    while (smz--)
      *pr++ = *plg++;

  for (;lgz && smu; --lgz, --smu)
  {
    diff -= *psm++;
    *pr++ = (Word) diff;
    diff >>= kWSC;
  }

  r.dNum->bpt = bpt;

  // If there are some large 0's left over, it means all of the small words are to
  // the right of all of the large words
  for (; lgz; --lgz)
    *pr++ = (Word) diff;

  // Subtract both operands, moving the binary point instead of writing least significant 0's
  while (pr == r.dNum->num && smu-- && i++ < ru)
  { 
    diff += (SDWord)*plg++ - *psm++;
    if (*pr = (Word) diff)
      ++pr;
    else
      ++r.dNum->bpt;
    diff >>= kWSC;
  }

  // Subtract both operands
  while (smu-- > 0 && i++ < ru)
  { 
    diff += (SDWord)*plg++ - *psm++;
    *pr++ = (Word) diff;
    diff >>= kWSC;
  }

  // Propagate borrow to larger operand, moving the binary point instead of writing least
  // significant 0's
  while (pr == r.dNum->num && i++ < ru)
  { 
    diff += *plg++;
    if (*pr = (Word) diff)
      ++pr;
    else
      ++r.dNum->bpt;
    diff >>= kWSC;
  }

  // Propagate borrow to larger operand
  while (i++ < ru)
  { 
    diff += *plg++;
    *pr++ = (Word) diff;
    diff >>= kWSC;
  }

  // Figure out how many words actually used
  while (!*--pr);
  r.dNum->nwu = (long)(pr - r.dNum->num) + 1;
  r.verify();
  return r;
}

Bignum operator - (const Bignum& a, const Bignum& b)
{
  Bignum r;
  return Bignum::bigsub (a, b, r);
}

/*      *********************
        *                   *
        *  Shift Operators  *
        *                   *
        *********************
*/
Bignum& Bignum::biglsh (const Bignum& a, long b, Bignum& r)
{ 
  if (!a.dSign || !b)
    return r = a;
  if (b < 0)
    return bigrsh (a, -b, r);

  // Split shift into words and bits
  long ws  = b / kWSC;				// Word shift
  int bsl = b % kWSC;				// bit shift left
  int bsr = kWSC - bsl;				// bit shift right

  // If all bits are shifted out of LSW, this becomes a shift right
  if ((Word)(a.dNum->num[0] << bsl) == 0)
  {
    bigrsh (a, bsr, r);
    r.addbpt (ws + 1);
    return r;
  }

  // How many words do we need?
  long au  = a.dNum->nwu;
    // This version doesn't work due to optimizer bug introduced in VStudio 2005. The compiler
    // doesn't cast to DWord, which is a valid optimization except for the case bsr == kWSC,
    // which is the only reason the cast is there in the first place. (27-Jul-2008)
    //int ovf = ((DWord)a.dNum->num[au - 1] >> bsr) != 0;
  int ovf = bsr < kWSC && (a.dNum->num[au - 1] >> bsr) != 0;	// some bits will be shifted out of MSW?
  long ru  = au + ovf;

  // Make result as appropriate	
  Bignum v = a;		// in case r is a
  r.allocate (ru, &r == &a);

  // do it low to high
  Word *pr = r.dNum->num;
  Word *pa = v.dNum->num;

  if (bsl)
  {
    DWord link = 0;
    while (au--)
    { 
      link |= (DWord)*pa++ << bsl;
      *pr++ = (Word) link;
      link >>= kWSC;
    }
    if (link)
      *pr = (Word) link;
  }
  else
    if (pr != pa)
      while (au--)
        *pr++ = *pa++;

  r.dSign = a.dSign;
  r.dNum->nwu = ru;
  r.dNum->bpt = v.dNum->bpt + ws;
  r.verify();
  return r;
}

Bignum& Bignum::bigrsh (const Bignum& a, long b, Bignum& r)
{ 
  if (!a.dSign || !b)
    return r = a;
  if (b < 0)
    return biglsh (a, -b, r);

  // Split shift into words and bits
  long ws  = b / kWSC;				// Word shift
  int bsr = b % kWSC;				// bit shift right
  int bsl = kWSC - bsr;				// bit shift left

  // if some bits are shifted out of LSW, this becomes a shift left
  if ((Word)((DWord)a.dNum->num[0] << bsl) != 0)
  {
    biglsh (a, bsl, r);
    r.addbpt (-(ws + 1));
    return r;
  }

  // How many words do we need?
  long au = a.dNum->nwu;
  int unf = (a.dNum->num[au - 1] >> bsr) == 0;	// all bits will be shifted out of MSW?
  long ru = au - unf;
  
  // Make result as appropriate
  Bignum v = a;		// in case r is a
  r.allocate (ru, &r == &a);

  // do it low to high
  Word *pr = r.dNum->num;
  Word *pa = v.dNum->num;

  if (bsr)
  {
    DWord link = *pa++ >> bsr;
    while (--au)
    { 
      link |= (DWord)*pa++ << bsl;
      *pr++ = (Word) link;
      link >>= kWSC;
    }
    if (link)
      *pr = (Word) link;
  }
  else
    if (pr != pa)
      while (au--)
        *pr++ = *pa++;


  r.dSign = a.dSign;
  r.dNum->nwu = ru;
  r.dNum->bpt = v.dNum->bpt - ws;
  r.verify();
  return r;
}

Bignum operator << (const Bignum& a, long b)
{ 
  Bignum r;
  return Bignum::biglsh (a, b, r);
}

Bignum operator >> (const Bignum& a, long b)
{ 
  Bignum r;
  return Bignum::bigrsh (a, b, r);
}

Bignum& Bignum::addbpt (long n)
{
  if (dSign && n)
  {
    if (dNum->refc > 1)
    {
      Bignum s = *this;
      allocate (dNum->nwu);
      for (long i = 0; i < s.dNum->nwu; ++i)
        dNum->num[i] = s.dNum->num[i];
      dNum->nwu = s.dNum->nwu;
      dNum->bpt = s.dNum->bpt + n;
    }
    else
      dNum->bpt += n;
  }

  return *this;
}

/*      *************************
        *                       *
        *  Truncate to Integer  *
        *                       *
        *************************
*/
Bignum Bignum::integer (bool round) const
{
  if (!dSign || dNum->bpt >= 0)
    return *this;

  // How many words do we need for the result?
  long ru = dNum->nwu + dNum->bpt;

  // Is there a carry due to rounding?
  long carry = round && ru >= 0 && dNum->num[-dNum->bpt - 1] & (1 << (kWSC - 1));

  if (ru + carry <= 0)
    return 0;

  // Make result
  Bignum r;
  r.allocate (ru + carry);
  r.dNum->bpt = 0;

  Word *pr = r.dNum->num;
  Word *pa = dNum->num - dNum->bpt;
  while (ru && !*pa)
  {
    ++pa;
    ++r.dNum->bpt;
    --ru;
  }

  for (long i = 0; i < ru; ++i)
    *pr++ = *pa++;

  r.dNum->nwu = ru;
  r.dSign = ru > 0;
  r.verify();
  if (carry)
    r += 1;
  r.dSign = dSign;

  return r;
}

/*      **************************
        *                        *
        *  Comparison Operators  *
        *                        *
        **************************

Bignum comparison utility. Returns 0 if a == b, a positive value if a > b,
and a negative value if a < b. If the signs of a and b are ==, then the
magnitude of the returned value is >= to the number of words that would
be required to store |a - b|; otherwise just return +1 or -1
*/

long Bignum::bigcmp (const Bignum& a, const Bignum& b)
{ 
  // case where one or both signs are 0, or signs are !=
  if (!a.dSign)
    return -b.dSign;
  if (!b.dSign)
    return a.dSign;
  if (a.dSign != b.dSign)
    return a.dSign;

  // Signs == and non-zero. get ready to to compare
  long ai  = a.dNum->nwu, bi  = b.dNum->nwu;
  long abp = a.dNum->bpt, bbp = b.dNum->bpt;
  Word *pa = a.dNum->num, *pb = b.dNum->num;

  // skip MS 0's (shouldn't be any, but...)
  while (ai && !pa[ai - 1])
    --ai;
  while (bi && !pb[bi - 1])
    --bi;

  // if one operand has more words, don't need to compare
  if (ai + abp > bi + bbp)
    return ai + abp;
  if (ai + abp < bi + bbp)
    return -(bi + bbp);

  // OK, now we really have to compare
  while (ai && bi)
    if (pa[--ai] != pb[--bi])
      return pa[ai] > pb[bi] ? ai + abp + 1 : -(bi + bbp + 1);

  while (ai)
    if (pa[--ai])
      return ai + abp + 1;

  while (bi)
    if (pb[--bi])
      return -(bi + bbp + 1);

  return 0;
}

/*      **************
        *            *
        *  Multiply  *
        *            *
        **************
*/

Bignum& Bignum::bigmul (const Bignum& a, const Bignum& b, Bignum& r)
{
  // shortcut for various identities
  if (!a.dSign || !b.dSign)
    r = Bignum();
  else if (a.dNum->nwu == 1 && a.dNum->num[0] == 1)
  {
    const Bignum m = a;	// in case r is a
    r = b;
    r.dSign *= m.dSign;
    r.addbpt (m.dNum->bpt);
  }
  else if (b.dNum->nwu == 1 && b.dNum->num[0] == 1)
  { 
    const Bignum m = b;	// in case r is b
    r = a; 
    r.dSign *= m.dSign;
    r.addbpt (m.dNum->bpt);
  }
  else
  {
    // Operand with fewest number of words is the multiplier, for speed. Make new
    // copies of the operands for the allocate call below
    Word *pa, *pb;
    long
      au = a.dNum->nwu,
      bu = b.dNum->nwu,
      ru = au + bu;
    if (au > bu)
    {
      au = b.dNum->nwu;
      bu = a.dNum->nwu;
      pa = b.dNum->num;
      pb = a.dNum->num;
    }
    else
    {
      pa = a.dNum->num;
      pb = b.dNum->num;
    }

    Bignum saver;
    int rSign = a.dSign * b.dSign;
    long bpt = a.dNum->bpt + b.dNum->bpt;
    r.allocate2(ru, saver, au > 1 && (a.dNum == r.dNum || b.dNum == r.dNum));
    r.dSign = rSign;
    r.dNum->bpt = bpt;

    bool first = true;
    bpt = 0;
    Word* pr = nullptr;
    for (long i = 0; i < au; ++i)
    { 
      Word m = *pa++;
      pr = r.dNum->num + i - bpt;
      if (m)
      {
#	if !P4_ASM
          DWord prod = 0;
          long j;
#	endif
        Word* pm = pb;
        if (first)
        {
#	  if !P4_ASM
            for (j = 0; j++ < bu; ++bpt)
            {
              prod += (DWord)m * *pm++;
              *pr = (Word) prod;
              prod >>= kWSC;
              if (*pr)
              {
                ++pr;
                break;
              }
            }
            for (; j < bu; ++j)
            { 
              prod += (DWord)m * *pm++;
              *pr++ = (Word) prod;
              prod >>= kWSC;
            }
#	  else
            __asm
            {
              // eax  arithmetic ops
              // edx  arithmetic ops
              // ebx  prod
              // ecx  loop counter, array index
              // esi  pm
              // edi  pr

              mov esi, pm
              mov ecx, bu
              mov edi, pr
              xor ebx, ebx
              mov edx, m
              mov eax, [esi]
              lea esi, [esi+ecx*4]
              lea edi, [edi+ecx*4]
              neg ecx
              jz quit1
  top1:
              mul edx
              add eax, ebx
              jnz enter2
              mov eax, [esi+ecx*4+4]
              adc edx, 0
              inc bpt
              sub edi, 4
              mov ebx, edx
              mov edx, m
              inc ecx
              jnz top1
              jmp done
  top2:
              mul edx
              add eax, ebx
  enter2:
              adc edx, 0
              mov [edi+ecx*4], eax
              mov eax, [esi+ecx*4+4]
              mov ebx, edx
              mov edx, m
              inc ecx
              jnz top2
  done:
              mov [edi], ebx
              add edi, 4
              mov pr, edi
  quit1:
            }
#	  endif
          first = false;
        }
        else
#	  if !P4_ASM
            for (j = 0; j < bu; ++j)
            { 
              prod += *pr + (DWord)m * *pm++;
              *pr++ = (Word) prod;
              prod >>= kWSC;
            }
        *pr++ = (Word) prod;
#	  else
            __asm
            {
              // eax  arithmetic ops
              // edx  arithmetic ops
              // ebx  prod
              // ecx  loop counter, array index
              // esi  pm
              // edi  pr

              mov esi, pm
              mov ecx, bu
              mov edi, pr
              xor ebx, ebx
              mov edx, m
              mov eax, [esi]
              lea esi, [esi+ecx*4]
              lea edi, [edi+ecx*4]
              neg ecx
              jz quit2
  top:
              mul edx
              add eax, ebx
              mov ebx, [edi+ecx*4]
              adc edx, 0
              add eax, ebx
              adc edx, 0
              mov [edi+ecx*4], eax
              mov eax, [esi+ecx*4+4]
              mov ebx, edx
              mov edx, m
              inc ecx
              jnz top

              mov [edi], ebx
              add edi, 4
              mov pr, edi
  quit2:
            }
#	  endif
      }
      else
        if (first)
          ++bpt;
        else
        {
          pr += bu;
          *pr++ = 0;
        }
    }

    while (pr > r.dNum->num && !pr[-1])
      --pr;
    r.dNum->nwu = (long)(pr - r.dNum->num);
    r.dNum->bpt += bpt;
  }

  r.verify();
  return r;
}


Bignum operator * (const Bignum& a, const Bignum& b)
{ 
  Bignum r;
  return Bignum::bigmul (a, b, r);
}


/*      ******************
        *                *
        *  Bit Counting  *
        *                *
        ******************
*/
long Bignum::bits () const
{ 
  if (!dSign)
    return 0;

  Word *p = dNum->num + dNum->nwu;
  while (!*--p);

  int i = 0;
  for (Word w = *p; w; w >>= 1)
    ++i;

  return ((long)(p - dNum->num) + dNum->bpt) * kWSC + i;
}

long Bignum::precision () const
{
  if (!dSign)
    return 0;

  Word* p;
  for (p = dNum->num; !*p; ++p);

  int i = 0;
  for (Word w = *p; w; w <<= 1)
    ++i;

  return bits() - (((long)(p - dNum->num) + dNum->bpt + 1) * kWSC - i);
}

/*      ************
        *          *
        *  Divide  *
        *          *
        ************
*/
Bignum& Bignum::bigdiv (const Bignum& a, const Bignum& b, Bignum &q, Bignum* rem)
{ 
  long qbits = a.bits () - b.bits ();
  if (qbits < 0 || !b.dSign)
  {
    if (rem)
      *rem = a;
    return q = Bignum();
  }

  if (b.dNum->nwu == 1 && b.dNum->num[0] == 1)
  {
    // divide by 1 case
    if (!b.dNum->bpt && a.dNum->bpt >= 0)
    {
      // do most likely (a integer, b = 1) case fast
      if (rem)
        *rem = Bignum();
      q = a;
      q.dSign *= b.dSign;
      return q;
    }

    // need qTemp in case q is a or b
    Bignum qTemp = Bignum(a).addbpt(-b.dNum->bpt).integer();
    if (rem)
      *rem = a - Bignum(qTemp).addbpt(b.dNum->bpt);
    qTemp.dSign *= b.dSign;
    q = qTemp;
    q.verify();
    return q;
  }

  int qsign = a.dSign * b.dSign, rsign = a.dSign;
  
  // set up dividend, divisor, quotient
  Bignum dd = a.abs();

  // if b is a single word, do long division
  if (b.dNum->nwu == 1)
  { 
    Word dvsr = b.dNum->num[0];
    long idd = dd.dNum->nwu;
    DWord d = dd.dNum->num[--idd];
    long rbpt = std::min (b.dNum->bpt, dd.dNum->bpt);
    
    long qu = qbits / kWSC + 1;

    if (d >= dvsr)
    {
      d = 0;
      ++idd;
    }
    else if (qbits % kWSC == 0)
      --qu;

    if (qu)
    {
      // Can destroy q even if q is a or b because we have dd and don't need dv
      q.allocate (qu, &q == &a, idd > qu);
      Word* pq = q.dNum->num + qu;
      q.dNum->nwu = qu;
      q.dNum->bpt = qu;

      // this is the divide loop
      while (qu--)
      {
        d <<= kWSC;
        if (idd)
          d += dd.dNum->num[--idd];
        *--pq = (Word) (d / dvsr);
        if (*pq)
          q.dNum->bpt = qu;
        d = d % dvsr;
      }

      // Set sign of result. Handle case where quotient is 0
      q.dSign = q.dNum->nwu ? qsign : 0;

      // If there were 0's in one or more LSWs, lose them
      if (q.dNum->bpt)
      {
        q.dNum->nwu -= q.dNum->bpt;
        for (int i = 0; i < q.dNum->nwu; ++i)
          q.dNum->num[i] = q.dNum->num[i + q.dNum->bpt];
      }

      // Compute the remainder. idd is >0 if there is a fractional part
      if (rem)
      {
        long du = idd + (d != 0);
        if (du)
        {
          rem->allocate (du);
          rem->dSign = rsign;
          rem->dNum->nwu = du;
          for (int i = 0; i < idd; ++i)
            rem->dNum->num[i] = dd.dNum->num[i];
          if (d)
            rem->dNum->num[idd] = (Word) d;
          rem->dNum->bpt = rbpt;
        }
        else
          *rem = Bignum();
      }
    }
    else
    {
      if (rem)
        *rem = a;
      q = Bignum();
    }

    if (rem)
      rem->verify();
    q.verify();
    return q;
  }

# ifndef NDEBUG
    Bignum savedd = dd;
# endif

  // "5" determined emperically by computing Bernoulli numbers (see numbers.cpp)
  // and picking the value with the fastest execution time
  if (qbits < 5)
  {
    // do it the slow way (shift and subtract)
    Bignum dv = b.abs() << qbits;
    Bignum one = 1;
  
    // handle first bit
    if (dv <= dd)
    {
      dd -= dv;
      q = one;
    }
    else
      q = Bignum();

    // main divide loop
    long shift = 0;
    while (qbits--)
    { 
      ++shift;
      dv >>= 1;
      if (dv <= dd)
      { 
        dd -= dv;
        q <<= shift;
        shift = 0;
        q += one;
      }
    }
    q <<= shift;
  }
  else
  {
    // Do it the fancy way, using a converging series and the hardware's ability
    // to approximate the reciprocal of the divisor very quickly. Each iteration
    // gets around N more bits of quotient, where N is the precision of long
    // doubles (around 53 bits for typical implementations). 
    long exp;
    Bignum dv = b.abs();
    Bignum v = 1 / dv.frexp<long double>(exp);
    v >>= exp;

    q = Bignum();
    Bignum dq;
    do
    {
      // "a" is a word shift count whose purpose is to select approximately
      // the most significant N bits of dd, because any more just wastes time
      // in the multiply that computing ddv
      long a = Math::iDiv (dd.bits() - (long)LDBL_MANT_DIG, (long)kWSC);
      dd.addbpt(-a);
      dq = dd.integer(true) * v;
      dd.addbpt(a);

      // "b" is a word shift count whose purpose is to select approximately
      // the most significant N bits of dq. The max call insures that dq
      // is an integer after the addbpt below.
      long b = std::max (Math::iDiv (dq.bits() - (long)LDBL_MANT_DIG, (long)kWSC), -a);
      dq.addbpt(-b);
      dq = dq.integer(true);

      if (dq == Bignum())
        break;
      dq.addbpt(a + b);

      q += dq;
      dd -= dq * dv;
    }
    while (dd.abs() >= dv);

    if (dd < 0)
    {
      dd += dv;
      q -= 1;
    }
    assert (savedd == dv * q + dd);
  }


  // Set signs, verify and return results
  if (q.dSign)
    q.dSign = qsign;
  if (dd.dSign)
    dd.dSign = rsign;
  if (rem)
  {
    *rem = dd;
    rem->verify();
  }
  q.verify();
  return q;
}

Bignum operator / (const Bignum& a, const Bignum& b)
{ 
  Bignum r;
  return Bignum::bigdiv (a, b, r, 0);
}

Bignum operator % (const Bignum& a, const Bignum& b)
{
  Bignum q, r;
  Bignum::bigdiv(a, b, q, &r);
  return r;
}

/*      ***********
        *         *
        *  Power  *
        *         *
        ***********
*/

Bignum Bignum::power (long n, Bignum mod) const
{
  Bignum p = *this;
  Bignum r;

  if (n >= 0)
  {
    r = 1;
    while (true)
    {
      if (n & 1)
      {
        r *= p;
        if (mod != Bignum())
          r = r % mod;
      }
      n >>= 1;
      if (n)
      {
        p *= p;
        if (mod != Bignum())
          p = p % mod;
      }
      else
        break;
    }
  }

  return r;
}

/*      *********
        *       *
        *  I/O  *
        *       *
        *********
*/

Bignum Bignum::read (const char*& p)
{
  Bignum r;
  bool neg = false;
  Bignum ten = 10;

  // skip whitespace
  while (std::strchr(" \t\r\n", *p))
    ++p;

  // save sign
  switch (*p)
  {
  case '-':
    neg = true;
    // fall through
  case '+':
    ++p;
  }

  // convert digits
  while (true)
  {
    int n = *p - '0';
    if (0 <= n && n < 10)
    {
      r *= ten;
      r += n;
      ++p;
    }
    else
      break;
  }

  // handle sign
  if (neg)
    r = -r;

  return r;
}

std::string Bignum::decimal (bool commas) const
{
  Bignum tens = commas ? 1000 : 10000;   // largest power of 10 that will fit in 1 Word
  const char* format = commas ? ",%03d" : "%04d";
  Bignum a = *this;
  std::string s, sign;

  if (a < 0)
    sign = "-", a = -a;
  while (true)
  {
    Bignum r;
    a.div(a, tens, r);
    if (a != Bignum())
      s = strFormat(format, r.makeint()) + s;
    else
    {
      s = strFormat("%d", r.makeint()) + s;
      break;
    }
  }
  return sign + s;
}

std::string Bignum::primeFact(bool showProgress) const
{
  static const int smallPrimes[4] = { 2, 3, 5, 7};
  static const int cycle[8] = { 4, 2, 4, 2, 4, 6, 2, 6};

  std::vector<unsigned long> primes;
  std::string s;

  unsigned long nextPrime;
  int phase = -4;
  Bignum x = *this;

  while (x > 1)
  {
    // compute the next prime
    if (phase < 0)
      nextPrime = smallPrimes[4 + phase++];
    else
    {
      unsigned int n = nextPrime;
      nextPrime = 0;
      while (!nextPrime)
      {
        n += cycle[phase];
        phase = (phase + 1) & 7;
        if (Bignum(n).power(2) > x)
          break;
        for (unsigned int i = 0; i < primes.size(); ++i)
        {
          unsigned long p = primes[i];
          if (n % p == 0) break;
          if (p * p >= n)
            nextPrime = n;
        }
      }
      if (!nextPrime)
        break;
    }
    if (phase >= 0)
      primes.push_back(nextPrime);

    Bignum p(nextPrime);
    int f = 0;
    while (true)
    {
      Bignum q, r;
      q.div(x, p, r);
      if (r != 0)
        break;
      ++f;
      x = q;
    }
    if (f)
    {
      if (s.length() > 0)
        s += " * ";
      s += strFormat("%d", nextPrime);
      if (f > 1)
        s += strFormat("^%d", f);
      if (showProgress)
        printf("%s\r", s.c_str());
    }
  }

  if (x != 1)
  {
    if (s.length() > 0)
      s += " * ";
    s += x.decimal();
  }

  return s;
}

std::string Bignum::hex (bool full) const
{
  const int digits = (kWSC + 3) >> 2;
  std::string s;
  if (dSign)
  {
    s += dSign > 0 ? "+" : "-";
    for (int i = dNum->nwu; i;)
    {
      if (i == -dNum->bpt)
        s += ".";
      else if (i != dNum->nwu)
        s += ",";
      s += strFormat ("%0*X", digits, dNum->num[--i]);
    }

    if (dNum->bpt > 0 || dNum->bpt < -dNum->nwu)
      s += strFormat ("e%+d", dNum->bpt);
  }
  else
    s = "0";
  if (full && dNum)
    s += strFormat ("[%d,%d]", dNum->refc, dNum->nwa);

  return s;
}

const char* Bignum::dump (bool full) const
{
  enum
  {
    kMaxWatch = 16,
    kMaxLen   = 256,
  };
  static struct
  {
    const void*   id;
    unsigned long lastUsed;
    char          display[kMaxLen];
  }
  displayTable[kMaxWatch];
  static unsigned long useTime;

  // Verify that I've been constructed, since this may be called from a watch
  // window before construction
  if (dSign > 1 || dSign < -1 || (dSign && !dNum) || 
      (dNum && (((unsigned long)(unsigned long long)dNum & 0xF0000003) != 0 ||
                dNum->refc < 1 || dNum->refc > 1000 ||
                dNum->nwa < 1 || dNum->nwu < 0 || dNum->nwu > 100000)))
    return "Apparently not constructed";

  
  // Find a table index
  unsigned long oldest = useTime;
  int oldIndex = -1;
  int index = 0;
  while (index < kMaxWatch && displayTable[index].id != this)
  {
    if (displayTable[index].lastUsed < oldest)
    {
      oldIndex = index;
      oldest = displayTable[index].lastUsed;
    }
    ++index;
  }
  if (index == kMaxWatch)
  {
    index = oldIndex;
    displayTable[index].id = this;
  }
  displayTable[index].lastUsed = ++useTime;

  // Set and return the display string
  strncpy (displayTable[index].display, hex(full).c_str(), kMaxLen - 1);
  return displayTable[index].display;
}

void Bignum::print_stats (std::ostream& out)
{
# if KEEP_STATS
  out << strFormat ("                 zeros     storage     bignums     largest\n");
  out << strFormat ("created   %12d%12d%12d%12d bits\n", dNZCr, dNBAl, dNBCr, dLBAl * kWSC);
  out << strFormat ("destroyed %12d%12d%12d\n", dNZDs, dNBDl, dNBDs);
# else
  out << "No statistics\n";
# endif
}


/*    **************************
      *                        *
      *  Performance Counters  *
      *                        *
      **************************
*/

PerformanceCounter::PerformanceCounter (const char* format)
: dFormat(format), dCount(0), dTotal(0)
{
}

PerformanceCounter::~PerformanceCounter()
{
  if (dCount)
          std::cout << strFormat (dFormat, dCount, dTotal, dTotal / dCount);
}

PerformanceCounter& PerformanceCounter::operator += (double x)
{
  ++dCount;
  dTotal += x;
  return *this;
}

PerformanceProbe::PerformanceProbe (PerformanceCounter& counter, long& stat)
: dCounter(counter), dStatistic(stat), dStartValue(stat)
{
}

PerformanceProbe::~PerformanceProbe ()
{
  dCounter += dStatistic - dStartValue;
}
