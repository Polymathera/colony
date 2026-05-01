use ${name_snake}::echo;

#[test]
fn smoke_round_trip() {
    assert_eq!(echo(42_i32), 42_i32);
}
