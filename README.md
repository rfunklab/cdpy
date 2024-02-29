# CDPY

A simple but fast implementation of the CD index and some variants for citation network analysis. This implementation trades memory for multiprocessing, using as many cores as the machine it's running on has available (less one, for politeness). 

This implementation can currently calculate the following disruption-like measures: 
- Citation count (i).
- [CD index (cd)](https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2366).
- [CD index without k-term (cdnok)](https://direct.mit.edu/qss/article/1/3/1242/96102/Are-disruption-index-indicators-convergently-valid)
- [CD index with j-type threshold (cd^j)](https://direct.mit.edu/qss/article/1/3/1242/96102/Are-disruption-index-indicators-convergently-valid)
- Backwards citations (bcites)
- I-type citations (icites)
- J-type citations (jcites)
- K-type citations (kcites)

The code assumes input data is given in a csv or parquet file which contains the citation network in **edge list format**. In particular, the code assumes the file has four columns:

- `citing_record_id`: the citing record identifier
- `cited_record_id`: the cited record identifier
- `citing_year`: the year of publication of the citing record
- `cited_year`: the year of publication of the cited record

The actual column names for `citing_record_id` and `cited_record_id` as well as their types may be changed by passing the appropriate arguments from the command line. However, `citing_year` and `cited_year` are required to be present in the input file under those names, as the CD index is technically only defined with respect to yearly data. 
See the `data/aps_2000-2005.csv.gz` file for an example citation network taken from [APS](https://journals.aps.org/datasets) between 2000 and 2005.  

The code supports any choice of time horizon that is greater than 0. 
These time horizons are recorded in the output data as values trailing the underscore. For example, `cd_5` corresponds to the CD index computed with the `--time_horizons` flag set to 5. 
Similarly `cdnok_25` is the CD index (without k term) calculated with a 25 year time horizon. Setting a time horizon that is larger than the entire time period spanned by the input data is equivalent to computing the all-time metrics. 

The script computes disruption values one year at a time. 
Disruption output values for each citation network element (e.g. paper) are saved out on a semi-regular basis by appending to an output file during execution. The frequency with which this file is appended (measured in years) to may be changed via the `--write_every` flag. 

## Requirements

This code was tested using Python 3.11 and depends on the following packages

- numpy
- pandas
- tqdm

These can be installed by running
```
pip install -r requirements.txt
```

## Usage

The code expects the `--citation_loc` flag to point to the citation network edge list file. 
The `--save_loc` flag dictates where the results files will be stored. 
The following command will compute the CD index on 5-year time horizon on the example APS citation network included in the `data` directory, saving the results to the same directory.

```
python cdindex.py --citation_loc data/aps_2000-2005.csv.gz --save_loc data --cd_types cd --time_horizons 5
```

The following command will compute the CD index without K-type citations on a 3-year time horizon, only counting J-type citations which cite at least 5 of the focal paper's citations.

```
python cdindex.py --citation_loc data/aps_2000-2005.csv.gz --save_loc data --cd_types cdnok --thresholds 5 --time_horizons 3
```

All of the `--cd_types`, `--thresholds`, and `--time_horizons` can take list arguments, potentially simplifying the need for multiple command line calls. 
