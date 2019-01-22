name=${1:-"*.eps"}
for f in $name
do
    #convert -density 600 $f ${f%.pdf}.png
    echo converting $f
    convert -density 450 $f ${f%.eps}.png
done
mv *.png png