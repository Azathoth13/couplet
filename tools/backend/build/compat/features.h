/* EasyFHE Ubuntu-22.04-compat overlay: parse the real glibc features.h, then
   turn off the ISO C2X strtol redirection so binaries built on glibc 2.38+
   do not require GLIBC_2.38 (__isoc23_strtol & co).  Everything else in
   _GNU_SOURCE stays enabled. */
#include_next <features.h>
#undef __GLIBC_USE_C2X_STRTOL
#define __GLIBC_USE_C2X_STRTOL 0
