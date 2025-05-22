

using StateSpaceEcon.Kalman

@isdefined(KF) || Core.eval(@__MODULE__, :(
    module KF
    using StateSpaceEcon
    using TimeSeriesEcon
    using StateSpaceEcon.Kalman
    struct KFTestModel
        NS::Int
        NO::Int
        islinear::Bool
    end
    model = KFTestModel(1, 1, true)
    Kalman.kf_length_x(m::KFTestModel) = m.NS
    Kalman.kf_length_y(m::KFTestModel) = m.NO
    Kalman.kf_is_linear(m::KFTestModel) = m.islinear
    function Kalman.kf_linear_model(model::KFTestModel)
        @assert model.islinear && (model.NS == model.NO == 1)
        m = KFLinearModel(model)
        m.mu .= [6]
        m.H .= [0.9;;]
        m.F .= [0.6;;]
        m.G .= [1;;]
        m.Q .= [0.7;;]
        m.R .= [0.8;;]
        return m
    end
    end # module 
))  # Core.eval 

@testset "KFData" begin

    for KFDTYPE in (:KFDataFilter, :KFDataFilterEx, :KFDataSmoother, :KFDataSmootherEx), RNG in (13, 1:13)
        @eval begin
            kfd = $KFDTYPE($RNG, KF.model)
            @test eltype(kfd) == Float64

            @test size(kfd.x0) == (1,)
            @test size(kfd.x_pred) == (1, 13)
            @test size(kfd.Px0) == (1, 1)
            @test size(kfd.Px_pred) == (1, 1, 13)
            @test size(kfd.y_pred) == (1, 13)
            @test size(kfd.Py_pred) == (1, 1, 13)
            @test size(kfd.loglik) == (13,)

            x0 = [0.0]
            @test @kfd_set!(kfd, 1, x0) == x0
            @test @kfd_get(kfd, 1, x0) == x0
            @test @kfd_view(kfd, 1, x0) == x0
            @test @kfd_view(kfd, 1, x0) isa SubArray

            x_pred = [1.0]
            @test @kfd_set!(kfd, 6, x_pred) == [1]
            @test @kfd_get(kfd, 6, x_pred) == [1]
            @test @kfd_view(kfd, 6, x_pred) isa SubArray
            @test (@kfd_view(kfd, 6, x_pred) .= 6; @kfd_get(kfd, 6, x_pred) == [6])
        end
    end

    k1 = KFDataFilter(1:12, KF.model)
    k2 = KFDataFilterEx(1:12, KF.model)
    for p in propertynames(k1)
        fill!(getproperty(k1, p), 3.0)
    end
    for p in propertynames(k2)
        fill!(getproperty(k2, p), 3.0)
    end
    @test @compare(k1, k1, quiet)
    @test @compare(k1, k2, ignoremissing, quiet)
    k1.x_pred[1] = 7
    @test @compare(k1, k2, ignoremissing, quiet) == false
end


@testset "KFAPI" begin

    @test kf_length_x(KF.model) == 1
    @test kf_length_y(KF.model) == 1
    @test kf_is_linear(KF.model) == true
    @test kf_linear_model(KF.model) isa KFLinearModel

    m = kf_linear_model(KF.model)
    @test kf_length_x(m) == 1
    @test kf_length_y(m) == 1
    @test kf_is_linear(m) == true
    @test m !== kf_linear_model(m)
end

@testset "KFilter" begin

    kd1 = KFDataFilter(1:12, KF.model)
    kf = KFilter(kd1)
    @test kf isa KFilter
    @test eltype(kf) == eltype(kd1)
    @test kf.kfd === kd1
    @test kf_length_x(kf) == kf_length_x(kd1) == 1
    @test kf_length_y(kf) == kf_length_y(kd1) == 1
    @test Kalman.kf_time_periods(kf) == 12

    kd12 = KFDataFilter(1:12, KF.KFTestModel(2,3,true))
    @test_throws r".*Cannot.*convert.*\{.*2, 3\}.*to.*\{.*1, 1\}.*"s  kf.kfd = kd12  # NS and NO dimensions don't match
    
    kd13 = KFDataFilter(Float32, 1:12, KF.model)
    @test_throws r".*Cannot.*convert.*\{Float32.*\}.*to.*\{Float64.*\}.*"s  kf.kfd = kd13  # Number types don't match
    
    kd2 = KFDataFilterEx(1:24, KF.model)
    @test (kf.kfd = kd2; kf.kfd === kd2)
    @test kf_length_x(kf) == kf_length_x(kd2) == 1
    @test kf_length_y(kf) == kf_length_y(kd2) == 1
    @test Kalman.kf_time_periods(kf) == 24

end




