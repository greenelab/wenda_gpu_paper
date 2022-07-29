# A master script that can be used to run all the simulated datasets
# on both wenda_orig and wenda_gpu, which is used for figure 1A in 
# the paper. Warning: running this script takes over a day.

# Run and time simulated datasets on wenda_gpu
bash time_wenda_gpu.sh simulated/sim_100_rep_0 100
bash time_wenda_gpu.sh simulated/sim_100_rep_1 100
bash time_wenda_gpu.sh simulated/sim_100_rep_2 100
bash time_wenda_gpu.sh simulated/sim_200_rep_0 100
bash time_wenda_gpu.sh simulated/sim_200_rep_1 100
bash time_wenda_gpu.sh simulated/sim_200_rep_2 100
bash time_wenda_gpu.sh simulated/sim_500_rep_0 100
bash time_wenda_gpu.sh simulated/sim_500_rep_1 100
bash time_wenda_gpu.sh simulated/sim_500_rep_2 100
bash time_wenda_gpu.sh simulated/sim_1000_rep_0 100
bash time_wenda_gpu.sh simulated/sim_1000_rep_1 100
bash time_wenda_gpu.sh simulated/sim_1000_rep_2 100
bash time_wenda_gpu.sh simulated/sim_1500_rep_0 100
bash time_wenda_gpu.sh simulated/sim_1500_rep_1 100
bash time_wenda_gpu.sh simulated/sim_1500_rep_2 100
bash time_wenda_gpu.sh simulated/sim_2000_rep_0 100
bash time_wenda_gpu.sh simulated/sim_2000_rep_1 100
bash time_wenda_gpu.sh simulated/sim_2000_rep_2 100
bash time_wenda_gpu.sh simulated/sim_5000_rep_0 100
bash time_wenda_gpu.sh simulated/sim_5000_rep_1 100
bash time_wenda_gpu.sh simulated/sim_5000_rep_2 100

# Run and time simulated datasets on wenda_gpu code but using CPUs
bash time_wenda_gpu_on_cpu.sh simulated/sim_100_rep_0 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_100_rep_1 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_100_rep_2 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_200_rep_0 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_200_rep_1 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_200_rep_2 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_500_rep_0 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_500_rep_1 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_500_rep_2 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_1000_rep_0 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_1000_rep_1 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_1000_rep_2 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_1500_rep_0 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_1500_rep_1 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_1500_rep_2 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_2000_rep_0 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_2000_rep_1 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_2000_rep_2 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_5000_rep_0 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_5000_rep_1 100
bash time_wenda_gpu_on_cpu.sh simulated/sim_5000_rep_2 100

# Run and time simulated datasets on original wenda software
bash time_wenda_orig.sh simulated/sim_100_rep_0
bash time_wenda_orig.sh simulated/sim_100_rep_1
bash time_wenda_orig.sh simulated/sim_100_rep_2
bash time_wenda_orig.sh simulated/sim_200_rep_0
bash time_wenda_orig.sh simulated/sim_200_rep_1
bash time_wenda_orig.sh simulated/sim_200_rep_2
bash time_wenda_orig.sh simulated/sim_500_rep_0
bash time_wenda_orig.sh simulated/sim_500_rep_1
bash time_wenda_orig.sh simulated/sim_500_rep_2
bash time_wenda_orig.sh simulated/sim_1000_rep_0
bash time_wenda_orig.sh simulated/sim_1000_rep_1
bash time_wenda_orig.sh simulated/sim_1000_rep_2
bash time_wenda_orig.sh simulated/sim_1500_rep_0
bash time_wenda_orig.sh simulated/sim_1500_rep_1
bash time_wenda_orig.sh simulated/sim_1500_rep_2
bash time_wenda_orig.sh simulated/sim_2000_rep_0
bash time_wenda_orig.sh simulated/sim_2000_rep_1
bash time_wenda_orig.sh simulated/sim_2000_rep_2
bash time_wenda_orig.sh simulated/sim_5000_rep_0
bash time_wenda_orig.sh simulated/sim_5000_rep_1
bash time_wenda_orig.sh simulated/sim_5000_rep_2
