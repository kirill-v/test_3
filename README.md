# filter-test
Usage: filter-test [params] image 

        -?, -h, --help (value:true)
                print this message
        --alpha (value:0.05)
                significance level from interval(0,1). Try pass (1-alpha) if result looks strange.
        --bins (value:64)
                number of bins in a histogram: 32, 64, 128
        --size (value:11)
                size of processing window

        image (value:<none>)
                input image

For example you can call it like:
`./filter-test --alpha=0.05 --bins=64 --size=13 ../../../lena.jpg`
