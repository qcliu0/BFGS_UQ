#!/bin/bash
for ii in {000..078}
do
	cd $ii
	#sed -s "s:    fname.*=.*:    fname = "${fname}":" ./plot_kernel*.py
	#sed -e "s#^fname.*#fname = $ii #g" ./plot_kernel*.py
	jj=$((10#$ii))
	echo $jj
	sed -i "s/tmp =.*/tmp = $jj/g" ./plot_kernel*.py
	sh ./ps.sh >/dev/null
	cd ..
done
