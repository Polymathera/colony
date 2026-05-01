#include <cassert>
#include "${name_snake}.hpp"

int main() {
    assert(${name_snake}::echo(0) == 0);
    assert(${name_snake}::echo(42) == 42);
    return 0;
}
