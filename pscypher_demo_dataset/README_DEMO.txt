PSCYPHER SYNTHETIC DEMO DATASET  -  every person, number and event is fictional.

CONTENTS
  town_10000_citizens.csv     10,000 unique fictional citizens (id, name, phone, address, occupation, ward)
  town_relationships.csv      13,614 background town ties (neighbours / acquaintances)
  FIR_01_SYNTHETIC_EXTORTION.pdf         complainant Nisha Bose
  FIR_02_SYNTHETIC_INVESTMENT_FRAUD.pdf  complainant Tanvi Rao
  FIR_03_SYNTHETIC_CARGO_THEFT.pdf       complainant Abhay Dubey
  cdr_500_demo.csv            exactly 500 call records  (calling person -> called person)
  financial_200_demo.csv      exactly 200 transactions  (sender -> recipient, amount_inr)
  ground_truth/ground_truth_DEMO_ONLY.csv        planted roles, FIR membership, core network, bridges
  ground_truth/ground_truth_links_DEMO_ONLY.csv  planted cross-case links
  (the ground_truth files are used only by the measured evaluation; the case loader never reads them)

QUICK LOAD
  Run the app from the repository root and use File > Load Demo Case. It finds this folder automatically.

MANUAL LOAD ORDER
  1. (optional, skips ~15 Yes/No identity prompts for manual uploads)
       Windows:  $env:PSCYPHER_AUTO_CONFIRM_EXACT="1"      Linux/macOS:  export PSCYPHER_AUTO_CONFIRM_EXACT=1
     Set it in the same terminal BEFORE launching the app. Safe here because every demo name is unique.
  2. Town Database > Import Town Citizens (CSV)       -> town_10000_citizens.csv
  3. Town Database > Import Town Relationships (CSV)  -> town_relationships.csv   (optional)
  4. File > Upload FIR  x3                            -> FIR_01, FIR_02, FIR_03 (works offline; no API call needed)
  5. File > Import Relationships (CSV)                -> select cdr_500_demo.csv AND financial_200_demo.csv together

EXPECTED RESULT (as documented by the dataset author; confirm against your own run)
  10 people, 65 distinct relationship edges (29 Calls + 36 Transactions), 700 observations, 0 rows skipped.
  Highest centrality: Rohan Mehta, Sahil Verma, Kavya Nair.
  Edge-weight anomaly on Rohan Mehta and Sahil Verma (planted: payments escalate about 3-4x in the last weeks).

THE 10 PEOPLE AND THE FIRs THEY APPEAR IN
  Nisha Bose      FIR-1 (complainant)
  Abhay Dubey     FIR-1 (witness), FIR-3 (complainant)
  Dev Saxena      FIR-1 (named caller)
  Sahil Verma     FIR-1, FIR-3
  Rohan Mehta     FIR-1, FIR-3
  Tanvi Rao       FIR-2 (complainant)
  Manav Joshi     FIR-2
  Ira Gupta       FIR-2
  Kavya Nair      FIR-2, FIR-3
  Aarav Iyer      FIR-3
  Cross-case link to find: FIR-1 and FIR-2 share no person, but call and payment records connect them
  through Rohan Mehta - Kavya Nair. Good question for the copilot: "Is there evidence connecting FIR-1 and FIR-2?"

MEASURED EVALUATION
  After loading everything: Analysis > Run Measured Evaluation (ground truth)... and pick
  ground_truth/ground_truth_DEMO_ONLY.csv (the links file must sit in the same folder, as it does here).
  Answer No to the LLM question for free graph-only metrics (FIR extraction, key players, cross-case ranking,
  link prediction). Answer Yes to compare models on the scored questions (uses Nebius credits; Q6 needs a Tavily key).
  Save as Markdown for the write-up.

NOTES
  - Uploading a FIR writes <name>_transcription.txt and a *_fir_cache_*.json beside the PDF (both are git-ignored).
    Delete those if you regenerate the PDFs.
  - The registry file pscypher_town.sqlite3 is created beside the app script (git-ignored). Delete it to start with an empty town.
  - The filename of each relationship CSV decides its label (names containing "cdr" -> Calls,
    "financial" -> Transactions), so keep those words in the filenames.
