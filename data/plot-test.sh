rgn=-R-5.05/-1.7/56.25/57.8 # Define Cairngorms
scl=-JM7.5c
projcent=-Jy-3.375/57.025/1:1 # Projection centre in Cairngorms

gmt makecpt -Cglobe -T-900/900 > topo.cpt

gmt grdimage topo.nc -R0/203400/0/172600 -JX7.5c -Ctopo.cpt -K -Baf > out.ps
awk -F',' '{print $1,$2}' fitted_samp_locs_geog.csv | gmt psxy -R0/203400/0/172600 -JX7.5c -Gblack -Sc0.1c -O >> out.ps

gmt psconvert out.ps -P -Tf -A0.2c
evince out.pdf

rm *.ps *.cpt
