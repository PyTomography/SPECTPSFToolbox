path=208keV_ME_PSF # signifies "medium energy" collimator

index_numbers=($(cat "positions.txt"))

for i in "${!index_numbers[@]}";
do
   # 53:1 and 59:1 are for full collimator modeling and random collimator shift. 01:208 means that the photon energy is 208keV. 26:20 is the number of photons, 12:${index_numbers[$i]} gives the radial position, and cc:sy-me is the collimator type.
  simind simind point_position${i}/53:1/59:1/01:208/26:20/12:${index_numbers[$i]}/cc:sy-me & 
done
wait

# Create a directory and move the output files to that directory
mkdir $path
mv point_position* $path