#pragma once

#define T_d

#ifdef T_d
typedef double current;
typedef double2 current2;
#else
typedef float current;
typedef float2 current2;
#endif // d
