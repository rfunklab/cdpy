import os
import argparse
import multiprocessing as mp
from time import time
from functools import partial
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np

TIME_HORIZONS = [5, 10, 20, 30, 300]
CD_TYPES = ["cd", "i", "cdnok", "bcites", "icites", "jcites", "kcites"]
THRESHOLDS = [1]
START_YEAR = None
START_YEAR_OUTPUT = None
END_YEAR = 2024
CITING_VAR = "citing_record_id"
CITED_VAR = "cited_record_id"
CITING_VAR_TYPE = "str"
CITED_VAR_TYPE = "str"
DATE_VAR_CITING = "citing_year"
DATE_VAR_CITED = "cited_year"
GARBAGE_COLLECT = False
DATE_TYPE = "year"
CITATION_LOC = "data/aps_2000-2005.csv"
SAVE_LOC = "data"
NWORKERS = max(1, mp.cpu_count() - 1)
WRITE_EVERY = 2
REREAD_CITATIONS = False
SHUFFLE = 0

quiet = False
citing_dict = defaultdict(set)
cited_dict = defaultdict(set)


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    global quiet

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        if not quiet:
            print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


@timer_func
def read_citation_csv(fileloc, low_memory=False, nrows=None, usecols=None):
    return pd.read_csv(
        fileloc,
        low_memory=low_memory,
        nrows=nrows,
        na_values="\\N",
        encoding_errors="ignore",
        on_bad_lines="warn",
        usecols=usecols,
    )


@timer_func
def read_citation_parquet(fileloc):
    return pd.read_parquet(fileloc)


def read_citations(fileloc):
    if ".csv" in fileloc:
        return read_citation_csv(fileloc)
    elif ".parquet" in fileloc:
        return read_citation_parquet(fileloc)
    else:
        raise ValueError(f"File type {fileloc} not supported")


@timer_func
def format_columns(df, date_type, date_var_citing, date_var_cited):
    # re-cast all column names into standard form
    df = df.rename(
        {date_var_citing: DATE_VAR_CITING, date_var_cited: DATE_VAR_CITED}, axis=1
    )
    if date_type == "date":
        df["citing_year"] = pd.to_datetime(df["citing_year"], errors="coerce")
        df["cited_year"] = pd.to_datetime(df["cited_year"], errors="coerce")
        df["citing_year"] = df["citing_year"].dt.year
        df["cited_year"] = df["cited_year"].dt.year
    elif date_type == "year":
        df["citing_year"] = df["citing_year"].apply(
            lambda x: int(x) if not np.isnan(x) else np.nan
        )
        df["cited_year"] = df["cited_year"].apply(
            lambda x: int(x) if not np.isnan(x) else np.nan
        )
    else:
        raise ValueError(f"Date type {date_type} not supported.")
    return df


def get_date_offset(date_type, time_horizon):
    return time_horizon


@timer_func
def set_citing_cited_dict(citations, citing_var, cited_var):
    global citing_dict
    global cited_dict
    citing_dict = (
        citations.groupby(citing_var)[cited_var].agg(set).to_dict(into=citing_dict)
    )
    cited_dict = (
        citations.groupby(cited_var)[citing_var].agg(set).to_dict(into=cited_dict)
    )


@timer_func
def pupdate_citing(citations, citing_var, cited_var):
    global citing_dict
    for k, v in citations.groupby(citing_var)[cited_var].agg(set).to_dict().items():
        citing_dict[k] |= v


@timer_func
def gc_citing(fids):
    # remove any citing papers less than the current year
    global citing_dict
    for k in fids:
        citing_dict.pop(k, None)


@timer_func
def pupdate_cited(citations, citing_var, cited_var):
    global cited_dict
    cited_dict = defaultdict(set)
    cited_dict = (
        citations.groupby(cited_var)[citing_var].agg(set).to_dict(into=cited_dict)
    )


@timer_func
def pupdate_cited_impact(citations, citing_var, cited_var):
    global citing_dict
    for k, v in citations.groupby(cited_var)[citing_var].agg(set).to_dict().items():
        cited_dict[k] |= v


def thresh_citing_cited(threshold, cited_dict, cited_by_f):
    if threshold > 1:
        cnts = [i for k in cited_by_f if k in cited_dict for i in cited_dict.get(k)]
        cnts = Counter(cnts)
        return set([k for k, v in cnts.items() if v >= threshold])
    return set().union(*[cited_dict.get(k, set()) for k in cited_by_f])


def icites(fix):
    cited_by_f = citing_dict[fix]
    citing_f = cited_dict.get(fix, set())

    citing_cited = thresh_citing_cited(1, cited_dict, cited_by_f)
    j_cites = citing_cited.intersection(citing_f)
    i_cites = citing_f.difference(j_cites)

    return {fix: len(i_cites)}


def jcites(threshold, fix):
    cited_by_f = citing_dict[fix]
    citing_f = cited_dict.get(fix, set())

    citing_cited = thresh_citing_cited(threshold, cited_dict, cited_by_f)
    j_cites = citing_cited.intersection(citing_f)

    return {fix: len(j_cites)}


def kcites(fix):
    cited_by_f = citing_dict[fix]
    citing_f = cited_dict.get(fix, set())

    citing_cited = thresh_citing_cited(1, cited_dict, cited_by_f)
    k_cites = citing_cited.difference(citing_f)

    return {fix: len(k_cites)}


def cd(threshold, fix):
    cited_by_f = citing_dict[fix]
    citing_f = cited_dict.get(fix, set())

    citing_cited = thresh_citing_cited(1, cited_dict, cited_by_f)
    j_cites = citing_cited.intersection(citing_f)
    k_cites = citing_cited.difference(citing_f)
    i_cites = citing_f.difference(j_cites)
    if threshold > 1:
        citing_cited = thresh_citing_cited(threshold, cited_dict, cited_by_f)
        j_cites = citing_cited.intersection(citing_f)
    try:
        cd = float(len(i_cites) - len(j_cites)) / float(
            len(i_cites) + len(j_cites) + len(k_cites)
        )
    except ZeroDivisionError:
        cd = np.nan
    return {fix: cd}


def cdnok(threshold, fix):
    cited_by_f = citing_dict[fix]
    citing_f = cited_dict.get(fix, set())

    citing_cited = thresh_citing_cited(1, cited_dict, cited_by_f)
    j_cites = citing_cited.intersection(citing_f)
    i_cites = citing_f.difference(j_cites)
    if threshold > 1:
        citing_cited = thresh_citing_cited(threshold, cited_dict, cited_by_f)
        j_cites = citing_cited.intersection(citing_f)
    try:
        cdnok = float(len(i_cites) - len(j_cites)) / float(len(i_cites) + len(j_cites))
    except ZeroDivisionError:
        cdnok = np.nan
    return {fix: cdnok}


def impact(fix):
    citing_f = cited_dict.get(fix, set())
    impact = len(citing_f)
    return {fix: impact}


def bcites(fix):
    cited_by_f = citing_dict.get(fix, set())
    bc = len(cited_by_f)
    return {fix: bc}


def get_function_partial(cd_type, threshold=1):
    if cd_type == "cd":
        return partial(cd, threshold)
    if cd_type == "cdnok":
        return partial(cdnok, threshold)
    if cd_type == "i":
        return impact
    if cd_type == "bcites":
        return bcites
    if cd_type == "icites":
        return icites
    if cd_type == "jcites":
        return partial(jcites, threshold)
    if cd_type == "kcites":
        return kcites
    else:
        raise ValueError(f"CD function type {cd_type} not supported.")


@timer_func
def read_citations_from_file(
    citation_loc,
    citing_var,
    cited_var,
    date_var_citing,
    date_var_cited,
    date_type,
    start_year,
    end_year,
):
    # read citations file and fix date column names
    citations = read_citations(citation_loc)
    assert (
        len(
            {citing_var, cited_var, date_var_citing, date_var_cited}.intersection(
                set(citations.columns)
            )
        )
        == 4
    ), "Citation file does not contain the necessary columns"

    citations = format_columns(citations, date_type, date_var_citing, date_var_cited)

    # set the start year and the first year to record disruption outputs
    start_year = (
        int(citations["citing_year"].min()) if start_year is None else start_year
    )

    # get full list of citations within this range
    all_citations = citations.loc[
        (citations["citing_year"] >= start_year)
        & (citations["citing_year"] <= end_year)
        & (citations[citing_var] != citations[cited_var])
        & (citations["citing_year"] >= citations["cited_year"])
    ]
    all_citations = all_citations.drop_duplicates(
        subset=[citing_var, cited_var], keep="first"
    )

    return all_citations, start_year


@timer_func
def save_out(years_res, save_loc, cd_col, start_year, prev_year, year):
    # save out intermediate file
    df = (
        pd.Series(years_res)
        .to_frame(cd_col)
        .reset_index()
        .rename({"index": "record_id"}, axis=1)
    )

    # update name of previous saved file, if exists
    savefile_temp_last = os.path.join(
        save_loc, f"{cd_col}_{start_year}-{prev_year}.csv.gz"
    )
    savefile_temp = os.path.join(save_loc, f"{cd_col}_{start_year}-{year}.csv.gz")
    try:
        os.rename(savefile_temp_last, savefile_temp)
    except OSError:
        pass

    df.to_csv(
        savefile_temp, index=False, header=not os.path.exists(savefile_temp), mode="a"
    )


def shuffle_citation_network(citation_df, cited_var):
    citation_df[cited_var] = citation_df.groupby(["cited_year", "citing_year"])[
        cited_var
    ].transform(np.random.permutation)
    return citation_df


def init(shared_citing_dict, shared_cited_dict):
    global citing_dict
    global cited_dict
    citing_dict = shared_citing_dict
    cited_dict = shared_cited_dict


def run(
    citation_loc=CITATION_LOC,
    save_loc=SAVE_LOC,
    cd_types=CD_TYPES,
    time_horizons=TIME_HORIZONS,
    thresholds=THRESHOLDS,
    start_year=START_YEAR,
    end_year=END_YEAR,
    start_year_output=START_YEAR_OUTPUT,
    citing_var=CITING_VAR,
    cited_var=CITED_VAR,
    citing_var_type=CITING_VAR_TYPE,
    cited_var_type=CITED_VAR_TYPE,
    garbage_collect=GARBAGE_COLLECT,
    reread_citations=REREAD_CITATIONS,
    date_var_citing=DATE_VAR_CITING,
    date_var_cited=DATE_VAR_CITED,
    date_type=DATE_TYPE,
    nworkers=NWORKERS,
    write_every=WRITE_EVERY,
    shuffle=SHUFFLE,
    suppress_logging=False,
):

    global quiet
    quiet = suppress_logging

    print("reading citation file...")
    all_citations, start_year = read_citations_from_file(
        citation_loc,
        citing_var,
        cited_var,
        date_var_citing,
        date_var_cited,
        date_type,
        start_year,
        end_year,
    )

    all_citations = all_citations.dropna(subset=[citing_var, cited_var])
    all_citations[citing_var] = all_citations[citing_var].astype(citing_var_type)
    all_citations[cited_var] = all_citations[cited_var].astype(cited_var_type)

    if shuffle > 0:
        np.random.seed(shuffle)
        all_citations = shuffle_citation_network(all_citations, cited_var)

    no_bcites = (
        all_citations[[cited_var, "cited_year"]]
        .drop_duplicates(subset=[cited_var])
        .set_index(cited_var)
    )
    no_bcites = no_bcites.loc[
        no_bcites.index.difference(all_citations[citing_var].unique())
    ]
    no_bcites = no_bcites.reset_index().set_index("cited_year")

    start_year_output = start_year if start_year_output is None else start_year_output

    for cd_type in cd_types:

        for time_horizon in time_horizons:

            for threshold in thresholds:

                global citing_dict
                global cited_dict
                citing_dict = defaultdict(set)
                cited_dict = defaultdict(set)

                cd_col = (
                    f"{cd_type}^{threshold}_{time_horizon}"
                    if threshold > 1
                    else f"{cd_type}_{time_horizon}"
                )

                # date offset to select only citations within the time horizon
                dlim = get_date_offset(date_type, time_horizon)

                if not os.path.exists(save_loc):
                    os.makedirs(save_loc)

                f = get_function_partial(cd_type, threshold=threshold)
                if cd_type == "bcites":
                    pupdate_citing(
                        all_citations[
                            all_citations["citing_year"] <= start_year_output + dlim
                        ],
                        citing_var,
                        cited_var,
                    )
                elif cd_type == "i":
                    pupdate_cited_impact(
                        all_citations[
                            all_citations["citing_year"] <= start_year_output + dlim
                        ],
                        citing_var,
                        cited_var,
                    )
                else:
                    set_citing_cited_dict(
                        all_citations[
                            all_citations["citing_year"] <= start_year_output + dlim
                        ],
                        citing_var,
                        cited_var,
                    )

                citation_years_actual = (
                    all_citations[all_citations["citing_year"] >= start_year_output][
                        "citing_year"
                    ]
                    .sort_values()
                    .unique()
                )
                citations = all_citations.set_index("citing_year")
                max_year = citations.index.max()
                print(
                    f"Computing CD Index type {cd_type} with time horizon {time_horizon} ({cd_col}):"
                )

                if reread_citations:
                    del all_citations

                years_res = {}
                for year in range(int(start_year_output), int(max_year + 1)):
                    if not quiet:
                        print(f"Running year: {year} with horizon: {year+dlim}...")

                    if year + dlim <= max_year:
                        ylim = year + dlim if year + dlim <= max_year else max_year
                        if ylim in citation_years_actual:
                            clim = citations.loc[ylim]
                            # if only one paper in the year, pandas returns a series
                            if isinstance(clim, pd.Series):
                                clim = clim.to_frame().T

                            if cd_type == "bcites":
                                pupdate_citing(clim, citing_var, cited_var)
                            elif cd_type == "i":
                                pupdate_cited_impact(clim, citing_var, cited_var)
                            else:
                                pupdate_citing(clim, citing_var, cited_var)
                                pupdate_cited(
                                    citations.loc[
                                        (citations.index.get_level_values(0) > year)
                                        & (citations.index.get_level_values(0) <= ylim)
                                    ],
                                    citing_var,
                                    cited_var,
                                )

                    if year in citation_years_actual:
                        if ylim not in citation_years_actual:
                            pupdate_cited(
                                citations.loc[
                                    (citations.index.get_level_values(0) > year)
                                    & (citations.index.get_level_values(0) <= ylim)
                                ],
                                citing_var,
                                cited_var,
                            )
                        fids = citations.loc[year, citing_var]
                        # if only one paper in the year, pandas returns a scalar
                        if isinstance(fids, pd.Series):
                            fids = fids.unique()
                        else:
                            fids = np.array([fids])

                        # now get any paper ids which do not make citations
                        bids = (
                            no_bcites.loc[year, cited_var]
                            if year in no_bcites.index
                            else pd.Series([], dtype=int)
                        )
                        # if only one paper in the year, pandas returns a scalar
                        if isinstance(bids, pd.Series):
                            bids = bids.unique()
                        else:
                            bids = np.array([bids])

                        fids = np.union1d(fids, bids)
                        res = {}
                        if nworkers > 1:
                            with mp.Pool(
                                nworkers,
                                initializer=init,
                                initargs=(citing_dict, cited_dict),
                            ) as pool:
                                for r in tqdm(
                                    pool.imap_unordered(f, fids),
                                    total=len(fids),
                                    disable=quiet,
                                ):
                                    res.update(r)
                        else:
                            for fix in tqdm(fids, disable=quiet):
                                res.update(f(fix))
                        years_res.update(res)

                        if garbage_collect:
                            gc_citing(fids)

                    if write_every > 0 and year % write_every == 0:
                        save_out(
                            years_res,
                            save_loc,
                            cd_col,
                            start_year,
                            year - write_every,
                            year,
                        )
                        # reset results to empty for next iteration
                        years_res = {}

                if len(years_res) > 0:
                    save_out(
                        years_res,
                        save_loc,
                        cd_col,
                        start_year,
                        year - (year % write_every),
                        year,
                    )
                if reread_citations:
                    all_citations, start_year = read_citations_from_file(
                        citation_loc,
                        citing_var,
                        cited_var,
                        date_var_citing,
                        date_var_cited,
                        date_type,
                        start_year,
                        end_year,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the CD index for a given citation file"
    )
    parser.add_argument(
        "--citation_loc",
        type=str,
        default=CITATION_LOC,
        help="Path to the citation data",
    )
    parser.add_argument(
        "--save_loc", type=str, default=SAVE_LOC, help="Location to output data"
    )
    parser.add_argument(
        "--cd_types",
        nargs="+",
        type=str,
        default=CD_TYPES,
        help="List of CD types to include",
    )
    parser.add_argument(
        "--time_horizons",
        nargs="+",
        type=int,
        default=TIME_HORIZONS,
        help="List of time horizons to compute",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=THRESHOLDS,
        help="List of jcite thresholds to compute",
    )
    parser.add_argument(
        "--start_year", type=int, default=START_YEAR, help="Start year of analysis"
    )
    parser.add_argument(
        "--end_year", type=int, default=END_YEAR, help="End year of analysis"
    )
    parser.add_argument(
        "--start_year_output",
        type=int,
        default=START_YEAR_OUTPUT,
        help="Start year for writing out analyses. Useful if process is stopped before full run completes",
    )
    parser.add_argument(
        "--citing_var",
        type=str,
        default=CITING_VAR,
        help="Name of the citing variable in the data",
    )
    parser.add_argument(
        "--cited_var",
        type=str,
        default=CITED_VAR,
        help="Name of the cited variable in the data",
    )
    parser.add_argument(
        "--citing_var_type",
        type=str,
        default=CITING_VAR_TYPE,
        help="Citing variable type",
    )
    parser.add_argument(
        "--cited_var_type", type=str, default=CITED_VAR_TYPE, help="Cited variable type"
    )
    parser.add_argument(
        "--garbage_collect",
        action="store_true",
        help="Whether to remove citing papers from memory after each year",
    )
    parser.add_argument(
        "--date_var_citing",
        type=str,
        default=DATE_VAR_CITING,
        help="Name of the citing date variable in the data",
    )
    parser.add_argument(
        "--date_var_cited",
        type=str,
        default=DATE_VAR_CITED,
        help="Name of the cited date variable in the data",
    )
    parser.add_argument(
        "--date_type",
        type=str,
        default=DATE_TYPE,
        help="Type of the date variable in the data",
        choices=["date", "year"],
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=NWORKERS,
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "--reread_citations",
        action="store_true",
        help="Whether to re-read citations between each centrality and horizon iteration",
    )
    parser.add_argument(
        "--shuffle",
        type=int,
        default=SHUFFLE,
        help="Whether to shuffle citation network before computing. Setting to value greater than 0 will run shuffle and set seed.",
    )
    parser.add_argument(
        "--write_every",
        type=int,
        default=WRITE_EVERY,
        help="How often to write output to disk",
    )
    parser.add_argument(
        "--suppress_logging",
        action="store_true",
        help="Whether to suppress logging output",
    )

    args = parser.parse_args()

    run(
        citation_loc=args.citation_loc,
        save_loc=args.save_loc,
        cd_types=args.cd_types,
        time_horizons=args.time_horizons,
        thresholds=args.thresholds,
        start_year=args.start_year,
        end_year=args.end_year,
        start_year_output=args.start_year_output,
        citing_var=args.citing_var,
        cited_var=args.cited_var,
        citing_var_type=args.citing_var_type,
        cited_var_type=args.cited_var_type,
        garbage_collect=args.garbage_collect,
        reread_citations=args.reread_citations,
        date_var_citing=args.date_var_citing,
        date_var_cited=args.date_var_cited,
        date_type=args.date_type,
        nworkers=args.nworkers,
        shuffle=args.shuffle,
        write_every=args.write_every,
        suppress_logging=args.suppress_logging,
    )
