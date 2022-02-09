from pathlib import Path
import csv
import re

from selenium import webdriver
import click
import numpy as np
import pandas as pd

from patents4IPPC import preprocessing


def scrape_patent_data(driver, patent_url, do_preprocess, use_titles):
    # Load the web page
    driver.get(patent_url)
    
    # Find abstract
    abstract_paragraphs = driver.find_elements_by_css_selector("abstract > div")
    abstract_text = " ".join([p.text for p in abstract_paragraphs])
    if do_preprocess:
        abstract_text = preprocessing.clean_helmers(abstract_text)

    # Optionally find title and prepend it to the abstract
    if use_titles:
        title = driver.find_element_by_css_selector("h1#title").text.strip()
        abstract_text = f"{title}. {abstract_text}"

    # Find claims
    def clean_claim(claim):
        claim = re.sub(r"^\d+\.\s*", "", claim)  # Remove claim number
        claim = re.sub(r"\s+", " ", claim)
        # ^ Replace newlines/tabs and multiple whitespaces with single 
        #   whitespace
        return claim
    claims = driver.find_elements_by_css_selector("div.claims > div")
    claims_texts_clean = [clean_claim(c.text) for c in claims]

    return abstract_text, claims_texts_clean

def save_qrel(
    query_patent_id,
    query_patent_abstract,
    query_patent_claims,
    resp_patent_id,
    resp_patent_abstract,
    resp_patent_claims,
    resp_patent_relevance,
    output_path,
    separate_sections    
):
    output_path = Path(output_path)
    if separate_sections:
        assert not output_path.exists() or output_path.is_dir(), \
            "Output path must be a directory."
        
        query_patent_file = output_path / "qs" / f"{query_patent_id}.dat"
        rel_patent_file = output_path / "rels" / f"{resp_patent_id}.dat"
        qrels_file = output_path / "qrels.txt"

        query_patent_file.parent.mkdir(parents=True, exist_ok=True)
        rel_patent_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(query_patent_file, "w") as fp:
            fp.writelines(
                "\n".join([query_patent_abstract] + query_patent_claims)
            )
        with open(rel_patent_file, "w") as fp:
            fp.writelines(
                "\n".join([resp_patent_abstract] + resp_patent_claims)
            )
        with open(qrels_file, "a+") as fp:
            fp.write(
                f"qs/{query_patent_id}.dat,"
                f"rels/{resp_patent_id}.dat,"
                f"{resp_patent_relevance}\n"
            )
    else:  # Save as normal csv
        if not output_path.exists():
            output_path.write_text(
                "query_id,query,response_id,response,label\n"
            )
        with open(output_path, mode="a") as fp:
            query_patent_content = " ".join(
                [query_patent_abstract] + query_patent_claims
            ).strip()
            resp_patent_content = " ".join(
                [resp_patent_abstract] + resp_patent_claims
            ).strip()

            writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
            writer.writerow([
                query_patent_id,
                query_patent_content,
                resp_patent_id,
                resp_patent_content,
                str(resp_patent_relevance)
            ])

@click.command()
@click.option(
    '-d', '--dataset-dir', 'path_to_dataset_dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help=('Path to a directory containing the Helmers manual dataset (with '
          '"human_eval" and "patent_sheets" as subdirectories).')
)
@click.option(
    '-p', '--preprocess', 'do_preprocess',
    is_flag=True,
    help='Do some preprocessing of the patent abstracts.'
)
@click.option(
    '-t', '--use-titles',
    is_flag=True,
    help=('Merge patent titles with their abstracts. Be aware that some of '
          'the titles in the Helmers dataset are truncated (they end with '
          'three dots (...)).')
)
@click.option(
    '-c', '--scrape-claims',
    is_flag=True,
    help=('In addition to abstract and possibly title, include claims as '
          'well. Note that this requires scraping them from Google Patents '
          'because the Helmers manual dataset contains only abstracts.')
)
@click.option(
    '-w', '--webdriver', 'path_to_webdriver',
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=('Path to a webdriver to be used by Selenium to scrape Google '
          'Patents. Required if --scrape-claims was specified, ignored '
          'otherwise.')
)
@click.option(
    '-wt', '--webdriver-type',
    type=click.Choice(['chrome', 'firefox']),
    default='chrome',
    help=('Webdriver type. Currently, only "chrome" or "firefox" are allowed. '
          'Ignored unless --scrape-claims was specified. Defaults to "chrome".')
)
@click.option(
    '-s', '--separate-sections',
    is_flag=True,
    help=('Keep the abstract and each individual claim separated from each '
          'other and save the dataset in a format that is suitable for use '
          'with a Hierarchical Transformer, i.e. in a directory structure '
          'rather than as a flat .csv file. Ignored unless --scrape-claims '
          'was specified.')
)
@click.option(
    '-st', '--add-section-tags',
    is_flag=True,
    help=('Prepend section tags ([abstract] and [claim]) to the text '
          'extracted from the respective section. Useful for use with '
          'bert-for-patents and derived models.')
)
@click.option(
    '-o', '--output-path',
    type=click.Path(),
    required=True,
    help='Path where the dataset will be saved.'
)
def main(
    path_to_dataset_dir,
    do_preprocess,
    use_titles,
    scrape_claims,
    path_to_webdriver,
    webdriver_type,
    separate_sections,
    add_section_tags,
    output_path
):
    dataset_dir = Path(path_to_dataset_dir)

    if scrape_claims:
        # Instantiate the web driver for scraping Google Patents
        if webdriver_type == "chrome":
            driver_options = webdriver.ChromeOptions()
            driver_options.headless = True
            driver = webdriver.Chrome(
                executable_path=path_to_webdriver, options=driver_options
            )
        elif webdriver_type == "firefox":
            driver_options = webdriver.FirefoxOptions()
            driver_options.headless = True
            driver = webdriver.Firefox(
                firefox_binary=path_to_webdriver,
                firefox_options=driver_options
            )
        else:
            raise ValueError(f"Unsupported webdriver type: {webdriver_type}.")
        
        # Scrape patents' abstract and claims
        patent_sheets_dir = dataset_dir / "patent_sheets"
        for annotations_file in patent_sheets_dir.iterdir():
            query_patent_id = annotations_file.stem.split("_")[0]
            query_patent_url = f"https:/www.google.de/patents/{query_patent_id}"
            query_patent_abstract, query_patent_claims = scrape_patent_data(
                driver, query_patent_url, do_preprocess, use_titles
            )

            if add_section_tags:
                query_patent_abstract = (
                    f"[abstract] {query_patent_abstract}"
                    if query_patent_abstract is not None else None
                )
                query_patent_claims = [
                    f"[claim] {claim}" for claim in query_patent_claims
                ]

            response_patents = (
                pd.read_csv(str(annotations_file), sep="\t")
                .dropna(subset=["human"])
            )
            for _, resp_patent in response_patents.iterrows():
                resp_patent_id = resp_patent["id"]
                resp_patent_abstract, resp_patent_claims = scrape_patent_data(
                    driver, resp_patent["link"], do_preprocess, use_titles
                )
                resp_patent_relevance = float(resp_patent["human"])

                if add_section_tags:
                    resp_patent_abstract = (
                        f"[abstract] {resp_patent_abstract}"
                        if resp_patent_abstract is not None else None
                    )
                    resp_patent_claims = [
                        f"[claim] {claim}" for claim in resp_patent_claims
                    ]
                
                save_qrel(
                    query_patent_id,
                    query_patent_abstract,
                    query_patent_claims,
                    resp_patent_id,
                    resp_patent_abstract,
                    resp_patent_claims,
                    resp_patent_relevance,
                    output_path,
                    separate_sections
                )

        driver.quit()
    else:        
        # Read patent pairs with human relevance judgments
        corpus_info_dir = dataset_dir / 'corpus_info'
        qrels_file = corpus_info_dir / 'human_label_pairs.npy'
        qrels = np.load(qrels_file, encoding='latin1', allow_pickle=True).item()
        
        # Read patent texts (title + abstract)
        patent_texts_file = corpus_info_dir / 'patcorpus_abstract.npy'
        patent_texts = np.load(
            patent_texts_file, encoding='latin1', allow_pickle=True
        ).item()
        
        # Optionally merge titles and abstracts
        def title_repl_func(matches):
            return matches[1] + '. ' if use_titles else ''
        patent_texts = {
            patent_id: re.sub(r'(.*?)\n', title_repl_func, patent_text, count=1)
            for patent_id, patent_text in patent_texts.items()
        }
        
        # Pre-process the abstracts, if requested
        if do_preprocess:
            patent_texts = {
                patent_id: preprocessing.clean_helmers(patent_text)
                for patent_id, patent_text in patent_texts.items()
            }
        
        # Build a dataframe with all the necessary information. Patent IDs
        # are converted to integers using the observation that they are of
        # the form "US123456789", so we just discard the first two
        # characters and convert the rest of the string to an integer
        df_helmers_manual = pd.DataFrame([
            (int(qid[2:]), patent_texts[qid], int(rid[2:]), patent_texts[rid],
            relevance_score)
            for (qid, rid), relevance_score in qrels.items()
        ], columns=['query_id', 'query', 'response_id' ,'response', 'label'])
        
        # Save the dataframe to disk
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_helmers_manual.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
