
@testset "SimData" begin
    @test_throws ArgumentError SimData(1M10, (:a, :b, :c), rand(10, 2))
    let nms = (:a, :b), dta = rand(20, 2), sd = SimData(2000Q1, nms, copy(dta)), dta2 = rand(size(dta)...)
        @test firstdate(sd) == 2000Q1
        @test lastdate(sd) == 2000Q1 + 20 - 1
        @test frequencyof(sd) == Quarterly
        @test sd isa AbstractMatrix
        @test size(sd) == size(dta)
        # integer indexing must be identical to dta
        for i in axes(dta, 2)
            @test sd[:,i] == dta[:,i]
        end
        for j in axes(dta, 1)
            @test sd[j,:] == dta[j,:]
        end
        for i in eachindex(dta)
            @test sd[i] == dta[i]
        end
        # set indexing with integers
        for i in eachindex(dta2)
            sd[i] = dta2[i]
        end
        @test sd[:] == dta2[:]
        for i in axes(dta, 1)
            sd[i,:] = dta[i,:]
        end
        @test sd[:] == dta[:]
        for i in axes(dta2, 2)
            sd[:,i] = dta2[:,i]
        end
        @test sd[:] == dta2[:]
        # access by dot notation (for the columns)
        @test propertynames(sd) == nms
        @test sd.a isa TSeries
        @test sd.b isa TSeries
        @test_throws BoundsError sd.c
        @test sd.a.values == dta2[:,1]
        sd.a[:] = dta[:,1]
        sd.b[:] = dta[:,2]
        @test sd[:] == dta[:]
        @test sd[:a] isa TSeries && sd[:a].values == dta[:,1]
        @test sd["b"] isa TSeries && sd["b"].values == dta[:,2]
        # 
        sd.a = dta[:,1]
        sd.b = dta[:,2]
        @test sd[:] == dta[:]
        # 
        sd[:] = dta[:]
        sd.a = sd.b
        @test sd[:,1] == sd[:,2]
        sd.a = zeros(size(dta, 1))
        @test sum(abs, sd.a.values) == 0
        @test_throws DimensionMismatch sd.a = ones(length(sd.a) + 5)
        # access to rows by MIT
        sd[:] = dta[:]
        @test sd[2000Q1] isa NamedTuple{nms,NTuple{length(nms),Float64}}
        for (i, idx) in enumerate(mitrange(sd))
            @test [values(sd[idx])...] == dta[i,:]
        end
        sd[2000Q1] = zeros(size(dta, 2))
        @test sum(abs, sd[1,:]) == 0
        @test_throws BoundsError sd[1999Q4]
        @test_throws DimensionMismatch sd[2000Q3] = zeros(size(dta, 2) + 1)
        sd[2000Q2] = (b = 7.0, a = 1.5)
        @test sd[2,:] == [1.5, 7.0]
        sd[2004Q4] = (a = 5.0,)
        @test sd[end,1] == 5.0
        @test_throws BoundsError sd[2004Q4] = (a = 5.0, c = 1.1)
        @test_throws ArgumentError sd[2000Y]
        @test_throws BoundsError sd[1999Q4] = (a = 5.0, b = 1.1)
        @test_throws BoundsError sd[2010Q4] = rand(2)
        # 
        @test sd[2001Q1:2002Q4] isa SimData
        @test sd[2001Q1:2002Q4] == sd[5:12,:]
        sd[2001Q1:2002Q4] = 1:16
        @test sd[5:12,:] == reshape(1.0:16.0, :, 2)
        @test_throws BoundsError sd[1111Q4:2002Q1]
        @test_throws BoundsError sd[2002Q1:2200Q2]
        @test_throws DimensionMismatch sd[2001Q1:2002Q4] = 1:17
        # assign new column
        sd1 = hcat(sd, c=sd.a + 3.0)
        @test sd1[nms] == sd
        @test sd1[(:a,:c)] == sd1[:,[1,3]]
        # access with 2 args MIT and Symbol
        @test sd[2001Q2, (:a, :b)] isa NamedTuple
        let foo = sd[2001Q2:2002Q1, (:a, :b)] 
            @test foo isa SimData
            @test size(foo) == (4, 2) 
            @test firstdate(foo) == 2001Q2
        end
        @test_throws BoundsError sd[1999Q1, (:a,)]
        @test_throws BoundsError sd[2001Q1:2001Q2, (:a, :c)]
        @test_throws Exception sd.c = 5
        @test similar(sd) isa typeof(sd)
        @test_throws BoundsError sd[1999Q1:2000Q4] = zeros(8, 2)
        @test_throws BoundsError sd[2004Q1:2013Q4, [:a, :b]]
        # setindex with two 
        sd[2001Q1, (:b,)] = 3.5
        @test sd[5,2] == 3.5
        sd[2000Q1:2001Q4, (:b,)] = 3.7
        @test all(sd[1:8,2] .== 3.7)
        @test_throws BoundsError sd[1999Q1:2000Q4, (:a, :b)] = 5.7
        @test_throws BoundsError sd[2000Q1:2000Q4, (:a, :c)] = 5.7
    end
end


@testset "SimData show" begin
    # test the case when column labels are longer than the numbers
    let io = IOBuffer(), sd =  SimData(1U, (:verylongandsuperboringnameitellya, :anothersuperlongnamethisisridiculous, :a), rand(20, 3) .* 100)
        show(io, sd)
        lines = readlines(seek(io, 0))
        # labels longer than 10 character are abbreviated with '…' at the end
        @test length(split(lines[2], '…')) == 3
    end
    let io = IOBuffer() , sd = SimData(1U, (:alpha, :beta), zeros(24, 2))
        show(io, sd)
        lines = readlines(seek(io, 0))
        lens = length.(lines)
        # when labels are longer than the numbers, the entire column stretches to fit the label
        @test lens[2] == lens[3]
    end
    nrow = 24
    letters = Symbol.(['a':'z'...])
    for (nlines, fd) = zip([3, 4, 5, 6, 7, 8, 22, 23, 24, 25, 26, 30], Iterators.cycle((qq(2010, 1), mm(2010, 1), yy(2010), ii(1))))
        for ncol = [2,5,10,20]
            # display size if nlines × 80
            # data size is nrow × ncol
            # when printing data there are two header lines - summary and column names
            io = IOBuffer()
            sd = SimData(fd, tuple(letters[1:ncol]...), rand(nrow, ncol))
            show(IOContext(io, :displaysize => (nlines, 80)), MIME"text/plain"(), sd)
            lines = readlines(seek(io, 0))
            @test length(lines) == max(3, min(nrow + 2, nlines - 3))
            @test maximum(length, lines) <= 80
            io = IOBuffer()
            show(IOContext(io, :limit => false), sd)
            lines = readlines(seek(io, 0))
            @test length(lines) == nrow + 2
        end
    end
end