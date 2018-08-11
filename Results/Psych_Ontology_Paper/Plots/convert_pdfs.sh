name=${1:-"*.pdf"}
for f in $name
do
    convert -density 300 $f ${f%.pdf}.png
done
mv *.png png