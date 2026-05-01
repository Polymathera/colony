using Test
using ${name_snake}

@testset "${name_snake} smoke" begin
    @test ${name_snake}.run(1) == 1
    @test ${name_snake}.run("x") == "x"
end
