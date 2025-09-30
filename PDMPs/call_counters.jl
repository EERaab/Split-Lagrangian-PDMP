using Dates

n_call_density = 0
n_call_grad = 0
n_call_hess = 0
n_call_dir_3 = 0
n_call_3 = 0
n_call_point = 0
time_density = 0
time_grad = 0
time_hess = 0
time_dir_3 = 0
time_3 = 0


function reset_call_counters()
    global n_call_density
    global n_call_grad
    global n_call_hess
    global n_call_dir_3
    global n_call_3
    global n_call_point
    global time_density
    global time_grad
    global time_hess
    global time_dir_3
    global time_3
    n_call_density = 0
    n_call_grad = 0
    n_call_hess = 0
    n_call_dir_3 = 0
    n_call_3 = 0
    n_call_point = 0
    time_density = 0
    time_grad = 0
    time_hess = 0
    time_dir_3 = 0
    time_3 = 0

end