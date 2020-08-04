
@testset "SimData" begin
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
        # 
        @test sd[2001Q1:2002Q4] isa SimData
        @test sd[2001Q1:2002Q4] == sd[5:12,:]
        sd[2001Q1:2002Q4] = 1:16
        @test sd[5:12,:] == reshape(1.0:16.0, :, 2)
        @test_throws DimensionMismatch sd[2001Q1:2002Q4] = 1:17
        # assign new column
        sd1 = hcat(sd, c=sd.a + 3.0)
        @test sd1[nms] == sd
        @test sd1[[:a,:c]] == sd1[:,[1,3]]
    end
end
