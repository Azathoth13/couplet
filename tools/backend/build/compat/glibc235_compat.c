/* Ubuntu 22.04 (glibc 2.35) compatibility shims for objects linked in from
 * libstdc++.a that was built against glibc 2.39 headers.
 *   __isoc23_strtoul  (GLIBC_2.38)  -> strtoul  (GLIBC_2.2.5)
 *   arc4random        (GLIBC_2.36)  -> getrandom (GLIBC_2.25)
 * Symbols are hidden so they never leak out of the .so that links this. */
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <sys/random.h>

#define HIDDEN __attribute__((visibility("hidden")))

HIDDEN unsigned long __isoc23_strtoul(const char *s, char **e, int b) { return strtoul(s, e, b); }
HIDDEN long __isoc23_strtol(const char *s, char **e, int b) { return strtol(s, e, b); }
HIDDEN long long __isoc23_strtoll(const char *s, char **e, int b) { return strtoll(s, e, b); }
HIDDEN unsigned long long __isoc23_strtoull(const char *s, char **e, int b) { return strtoull(s, e, b); }

HIDDEN uint32_t arc4random(void) {
  uint32_t v = 0;
  for (;;) {
    ssize_t n = getrandom(&v, sizeof v, 0);
    if (n == (ssize_t)sizeof v) return v;
    if (n < 0 && errno != EINTR && errno != EAGAIN) return v ^ (uint32_t)(uintptr_t)&v;  /* degrade, never hang */
  }
}
