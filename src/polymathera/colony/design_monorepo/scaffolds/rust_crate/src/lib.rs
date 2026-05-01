//! ${name} — ${description}
//!
//! Scaffolded into tools/${purpose}/${name}/ of a Polymathera Colony
//! design monorepo. Replace the smoke surface with the tool's real API.

/// Smoke entry point: returns the input value unchanged.
pub fn echo<T>(x: T) -> T {
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        assert_eq!(echo(1), 1);
        assert_eq!(echo("x"), "x");
    }
}
