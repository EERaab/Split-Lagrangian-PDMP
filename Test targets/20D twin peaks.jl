function log_twin_peaks_20D(X::Vector)
    sum1 = sum(X.^2)
    sum2 = sum((X .-0.5).^2)
    return log(exp(-sum1/2.0)+exp(-sum2/2.0)) - 10*log(2*π)
end

function normal(μ::Float64, C::Float64)
    return X -> (sqrt(2*π)^length(μ)*sqrt(C))^(-1)*exp(-((X-μ)^2)/(2*C))
end

x_marginal_distr(x) = (normal(0.0, 1.0)(x)+normal(0.5, 1.0)(x))/2
y_marginal_distr(y) = (normal(0.0, 1.0)(y)+normal(0.5, 1.0)(y))/2
    
target = TargetData(log_twin_peaks_20D, 20)