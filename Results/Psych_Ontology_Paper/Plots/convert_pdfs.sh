name=${1:-"*.pdf"}
for f in $name
do
    #convert -density 600 $f ${f%.pdf}.png
    echo converting $f
    pdftoppm -r 600 $f ${f%.pdf} -png
done
mv *.png png


