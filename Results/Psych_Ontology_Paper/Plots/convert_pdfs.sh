for f in *.pdf
do
    convert -density 300 $f ${f%.pdf}.png
done
mv *.png png