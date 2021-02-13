set term png size 1200,400

set output "image_rho_kappa_mu_kernels.png"

#set palette defined ( 1000 "blue", 3500 "yellow", 6000 "red")
set pm3d map
unset xtics
unset ytics
unset key
unset grid
set samples 5
#set cbrange [1000:6000]
set isosamples 5

set multiplot

set size 0.333,1
set origin 0,0
set title "rho"
splot "./proc000000_rho_kappa_mu_kernel.dat" using 1:2:3 w points palette pt 7 ps 0.5

set size 0.333,1
set origin 0.33,0
set title "inverted vp"
splot "./proc000000_rho_kappa_mu_kernel.dat" using 1:2:4 w points palette pt 7 ps 0.5

set size 0.333,1
set origin 0.64,0
set title "inverted vs"
splot "./proc000000_rho_kappa_mu_kernel.dat" using 1:2:5 w points palette pt 7 ps 0.5

unset multiplot


