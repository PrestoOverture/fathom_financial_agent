import os
import pandas as pd
from openai import OpenAI
from pandas import DataFrame
from dotenv import load_dotenv
from pydantic import BaseModel, Field

class QuestionAudit(BaseModel):
    label: str = Field(description="'simple retrieval' or 'reasoning'")
    reasoning: str = Field(description="Explain why this question is a 'simple retrieval' or 'reasoning' question.")

def audit_question(openai: OpenAI, question: str, answer: str, evidence: str) -> QuestionAudit:
    system_prompt = f"""
        You are a financial data auditor. Your job is to classify financial
        Q&A pairs as either 'simple_retrieval' or 'reasoning'. \n

        Definitions:
        - A question is 'simple_retrieval' if the answer can be obtained by
        directly copying a single fact (a number or phrase) from one place
        in the evidence, without any calculations, comparisons, or multi-step logic.
        - A question is 'reasoning' if an analyst must perform one or more of:
        * numerical calculations (e.g., differences, ratios, percentages),
        * comparisons across multiple periods, segments, or metrics,
        * multi-step logical reasoning that combines several pieces of evidence.\n

        The length of the evidence does NOT matter. Only the amount of thinking
        needed to go from the evidence to the answer matters.
    """

    simple_retrieval_evidence = "[{'evidence_text': 'Table of Contents \n3M Company and Subsidiaries\nConsolidated Statement of Cash Flow s\nYears ended December 31\n \n(Millions)\n \n2018\n \n2017\n \n2016\n \nCash Flows from Operating Activities\n \n \n \n \n \n \n \nNet income including noncontrolling interest\n \n$\n5,363 \n$\n4,869 \n$\n5,058 \nAdjustments to reconcile net income including noncontrolling interest to net cash\nprovided by operating activities\n \n \n \n \n \n \n \nDepreciation and amortization\n \n \n1,488 \n \n1,544 \n \n1,474 \nCompany pension and postretirement contributions\n \n \n(370) \n \n(967) \n \n(383) \nCompany pension and postretirement expense\n \n \n410 \n \n334 \n \n250 \nStock-based compensation expense\n \n \n302 \n \n324 \n \n298 \nGain on sale of businesses\n \n \n(545) \n \n(586) \n \n(111) \nDeferred income taxes\n \n \n(57) \n \n107 \n \n 7 \nChanges in assets and liabilities\n \n \n \n \n \n \n \nAccounts receivable\n \n \n(305) \n \n(245) \n \n(313) \nInventories\n \n \n(509) \n \n(387) \n \n57 \nAccounts payable\n \n \n408 \n \n24 \n \n148 \nAccrued income taxes (current and long-term)\n \n \n134 \n \n967 \n \n101 \nOther net\n \n \n120 \n \n256 \n \n76 \nNet cash provided by (used in) operating activities\n \n \n6,439 \n \n6,240 \n \n6,662 \n \n \n \n \n \n \n \n \nCash Flows from Investing Activities\n \n \n \n \n \n \n \nPurchases of property, plant and equipment (PP&E)\n \n \n(1,577) \n \n(1,373) \n \n(1,420) \nProceeds from sale of PP&E and other assets\n \n \n262 \n \n49 \n \n58 \nAcquisitions, net of cash acquired\n \n \n13 \n \n(2,023) \n \n(16) \nPurchases of marketable securities and investments\n \n \n(1,828) \n \n(2,152) \n \n(1,410) \nProceeds from maturities and sale of marketable securities and investments\n \n \n2,497 \n \n1,354 \n \n1,247 \nProceeds from sale of businesses, net of cash sold\n \n \n846 \n \n1,065 \n \n142 \nOther net\n \n \n 9 \n \n(6) \n \n(4) \nNet cash provided by (used in) investing activities\n \n \n222 \n \n(3,086) \n \n(1,403) \n \n \n \n \n \n \n \n \nCash Flows from Financing Activities\n \n \n \n \n \n \n \nChange in short-term debt net\n \n \n(284) \n \n578 \n \n(797) \nRepayment of debt (maturities greater than 90 days)\n \n \n(1,034) \n \n(962) \n \n(992) \nProceeds from debt (maturities greater than 90 days)\n \n \n2,251 \n \n1,987 \n \n2,832 \nPurchases of treasury stock\n \n \n(4,870) \n \n(2,068) \n \n(3,753) \nProceeds from issuance of treasury stock pursuant to stock option and benefit plans\n \n \n485 \n \n734 \n \n804 \nDividends paid to shareholders\n \n \n(3,193) \n \n(2,803) \n \n(2,678) \nOther net\n \n \n(56) \n \n(121) \n \n(42) \nNet cash provided by (used in) financing activities\n \n \n(6,701) \n \n(2,655) \n \n(4,626) \n \n \n \n \n \n \n \n \nEffect of exchange rate changes on cash and cash equivalents\n \n \n(160) \n \n156 \n \n(33) \n \n \n \n \n \n \n \n \nNet increase (decrease) in cash and cash equivalents\n \n \n(200) \n \n655 \n \n600 \nCash and cash equivalents at beginning of year\n \n \n3,053 \n \n2,398 \n \n1,798 \nCash and cash equivalents at end of period\n \n$\n2,853 \n$\n3,053 \n$\n2,398 \n \nThe accompanying Notes to Consolidated Financial Statements are an integral part of this statement.\n \n60', 'doc_name': '3M_2018_10K', 'evidence_page_num': 59, 'evidence_text_full_page': 'Table of Contents \n3M Company and Subsidiaries\nConsolidated Statement of Cash Flow s\nYears ended December 31\n \n(Millions)\n \n2018\n \n2017\n \n2016\n \nCash Flows from Operating Activities\n \n \n \n \n \n \n \nNet income including noncontrolling interest\n \n$\n5,363 \n$\n4,869 \n$\n5,058 \nAdjustments to reconcile net income including noncontrolling interest to net cash\nprovided by operating activities\n \n \n \n \n \n \n \nDepreciation and amortization\n \n \n1,488 \n \n1,544 \n \n1,474 \nCompany pension and postretirement contributions\n \n \n(370) \n \n(967) \n \n(383) \nCompany pension and postretirement expense\n \n \n410 \n \n334 \n \n250 \nStock-based compensation expense\n \n \n302 \n \n324 \n \n298 \nGain on sale of businesses\n \n \n(545) \n \n(586) \n \n(111) \nDeferred income taxes\n \n \n(57) \n \n107 \n \n 7 \nChanges in assets and liabilities\n \n \n \n \n \n \n \nAccounts receivable\n \n \n(305) \n \n(245) \n \n(313) \nInventories\n \n \n(509) \n \n(387) \n \n57 \nAccounts payable\n \n \n408 \n \n24 \n \n148 \nAccrued income taxes (current and long-term)\n \n \n134 \n \n967 \n \n101 \nOther net\n \n \n120 \n \n256 \n \n76 \nNet cash provided by (used in) operating activities\n \n \n6,439 \n \n6,240 \n \n6,662 \n \n \n \n \n \n \n \n \nCash Flows from Investing Activities\n \n \n \n \n \n \n \nPurchases of property, plant and equipment (PP&E)\n \n \n(1,577) \n \n(1,373) \n \n(1,420) \nProceeds from sale of PP&E and other assets\n \n \n262 \n \n49 \n \n58 \nAcquisitions, net of cash acquired\n \n \n13 \n \n(2,023) \n \n(16) \nPurchases of marketable securities and investments\n \n \n(1,828) \n \n(2,152) \n \n(1,410) \nProceeds from maturities and sale of marketable securities and investments\n \n \n2,497 \n \n1,354 \n \n1,247 \nProceeds from sale of businesses, net of cash sold\n \n \n846 \n \n1,065 \n \n142 \nOther net\n \n \n 9 \n \n(6) \n \n(4) \nNet cash provided by (used in) investing activities\n \n \n222 \n \n(3,086) \n \n(1,403) \n \n \n \n \n \n \n \n \nCash Flows from Financing Activities\n \n \n \n \n \n \n \nChange in short-term debt net\n \n \n(284) \n \n578 \n \n(797) \nRepayment of debt (maturities greater than 90 days)\n \n \n(1,034) \n \n(962) \n \n(992) \nProceeds from debt (maturities greater than 90 days)\n \n \n2,251 \n \n1,987 \n \n2,832 \nPurchases of treasury stock\n \n \n(4,870) \n \n(2,068) \n \n(3,753) \nProceeds from issuance of treasury stock pursuant to stock option and benefit plans\n \n \n485 \n \n734 \n \n804 \nDividends paid to shareholders\n \n \n(3,193) \n \n(2,803) \n \n(2,678) \nOther net\n \n \n(56) \n \n(121) \n \n(42) \nNet cash provided by (used in) financing activities\n \n \n(6,701) \n \n(2,655) \n \n(4,626) \n \n \n \n \n \n \n \n \nEffect of exchange rate changes on cash and cash equivalents\n \n \n(160) \n \n156 \n \n(33) \n \n \n \n \n \n \n \n \nNet increase (decrease) in cash and cash equivalents\n \n \n(200) \n \n655 \n \n600 \nCash and cash equivalents at beginning of year\n \n \n3,053 \n \n2,398 \n \n1,798 \nCash and cash equivalents at end of period\n \n$\n2,853 \n$\n3,053 \n$\n2,398 \n \nThe accompanying Notes to Consolidated Financial Statements are an integral part of this statement.\n \n60\n \n'}]"
    reasoning_evidence = "[{'evidence_text': 'Table of Contents\nACTIVISION BLIZZARD, INC. AND SUBSIDIARIES\nCONSOLIDATED BALANCE SHEETS\n(Amounts in millions, except share data)\n \nAt December 31, 2019\n \nAt December 31, 2018\nAssets\n \n \nCurrent assets:\n \n \nCash and cash equivalents\n$\n5,794\n $\n4,225\nAccounts receivable, net of allowances of $132 and $190, at December 31, 2019 and December 31, 2018, respectively\n848\n \n1,035\nInventories, net\n32\n \n43\nSoftware development\n322\n \n264\nOther current assets\n296\n \n539\nTotal current assets\n7,292\n \n6,106\nSoftware development\n54\n \n65\nProperty and equipment, net\n253\n \n282\nDeferred income taxes, net\n1,293\n \n458\nOther assets\n658\n \n482\nIntangible assets, net\n531\n \n735\nGoodwill\n9,764\n \n9,762\nTotal assets\n$\n19,845\n $\n17,890\n \n \n \nLiabilities and Shareholders Equity\n \n \nCurrent liabilities:\n \n \nAccounts payable\n$\n292\n $\n253\nDeferred revenues\n1,375\n \n1,493\nAccrued expenses and other liabilities\n1,248\n \n896\nTotal current liabilities\n2,915\n \n2,642\nLong-term debt, net\n2,675\n \n2,671\nDeferred income taxes, net\n505\n \n18\nOther liabilities\n945\n \n1,167\nTotal liabilities\n7,040\n \n6,498\nCommitments and contingencies (Note 23)\n \nShareholders equity:\n \n \n \nCommon stock, $0.000001 par value, 2,400,000,000 shares authorized, 1,197,436,644 and 1,192,093,991 shares issued at\nDecember 31, 2019 and December 31, 2018, respectively\n\n \n\nAdditional paid-in capital\n11,174\n \n10,963\nLess: Treasury stock, at cost, 428,676,471 shares at December 31, 2019 and December 31, 2018\n(5,563) \n(5,563)\nRetained earnings\n7,813\n \n6,593\nAccumulated other comprehensive loss\n(619) \n(601)\nTotal shareholders equity\n12,805\n \n11,392\nTotal liabilities and shareholders equity\n$\n19,845\n $\n17,890\nThe accompanying notes are an integral part of these Consolidated Financial Statements.\nF-4', 'doc_name': 'ACTIVISIONBLIZZARD_2019_10K', 'evidence_page_num': 68, 'evidence_text_full_page': 'Table of Contents\nACTIVISION BLIZZARD, INC. AND SUBSIDIARIES\nCONSOLIDATED BALANCE SHEETS\n(Amounts in millions, except share data)\n \nAt December 31, 2019\n \nAt December 31, 2018\nAssets\n \n \nCurrent assets:\n \n \nCash and cash equivalents\n$\n5,794\n $\n4,225\nAccounts receivable, net of allowances of $132 and $190, at December 31, 2019 and December 31, 2018, respectively\n848\n \n1,035\nInventories, net\n32\n \n43\nSoftware development\n322\n \n264\nOther current assets\n296\n \n539\nTotal current assets\n7,292\n \n6,106\nSoftware development\n54\n \n65\nProperty and equipment, net\n253\n \n282\nDeferred income taxes, net\n1,293\n \n458\nOther assets\n658\n \n482\nIntangible assets, net\n531\n \n735\nGoodwill\n9,764\n \n9,762\nTotal assets\n$\n19,845\n $\n17,890\n \n \n \nLiabilities and Shareholders Equity\n \n \nCurrent liabilities:\n \n \nAccounts payable\n$\n292\n $\n253\nDeferred revenues\n1,375\n \n1,493\nAccrued expenses and other liabilities\n1,248\n \n896\nTotal current liabilities\n2,915\n \n2,642\nLong-term debt, net\n2,675\n \n2,671\nDeferred income taxes, net\n505\n \n18\nOther liabilities\n945\n \n1,167\nTotal liabilities\n7,040\n \n6,498\nCommitments and contingencies (Note 23)\n \nShareholders equity:\n \n \n \nCommon stock, $0.000001 par value, 2,400,000,000 shares authorized, 1,197,436,644 and 1,192,093,991 shares issued at\nDecember 31, 2019 and December 31, 2018, respectively\n\n \n\nAdditional paid-in capital\n11,174\n \n10,963\nLess: Treasury stock, at cost, 428,676,471 shares at December 31, 2019 and December 31, 2018\n(5,563) \n(5,563)\nRetained earnings\n7,813\n \n6,593\nAccumulated other comprehensive loss\n(619) \n(601)\nTotal shareholders equity\n12,805\n \n11,392\nTotal liabilities and shareholders equity\n$\n19,845\n $\n17,890\nThe accompanying notes are an integral part of these Consolidated Financial Statements.\nF-4\n'}, {'evidence_text': 'Table of Contents\nACTIVISION BLIZZARD, INC. AND SUBSIDIARIES\nCONSOLIDATED STATEMENTS OF OPERATIONS\n(Amounts in millions, except per share data)\n \nFor the Years Ended December 31,\n \n2019\n \n2018\n \n2017\nNet revenues\n \n \n \n \nProduct sales\n$\n1,975\n $\n2,255 $\n2,110\nSubscription, licensing, and other revenues\n4,514\n \n5,245 \n4,907\nTotal net revenues\n6,489\n \n7,500 \n7,017\n \n \n \n \nCosts and expenses\n \n \n \n \nCost of revenuesproduct sales:\n \n \n \nProduct costs\n656\n \n719 \n733\nSoftware royalties, amortization, and intellectual property licenses\n240\n \n371 \n300\nCost of revenuessubscription, licensing, and other revenues:\n \n \n \nGame operations and distribution costs\n965\n \n1,028 \n984\nSoftware royalties, amortization, and intellectual property licenses\n233\n \n399 \n484\nProduct development\n998\n \n1,101 \n1,069\nSales and marketing\n926\n \n1,062 \n1,378\nGeneral and administrative\n732\n \n822 \n745\nRestructuring and related costs\n132\n \n10 \n15\nTotal costs and expenses\n4,882\n \n5,512 \n5,708\n \n \n \n \nOperating income\n1,607\n \n1,988 \n1,309\nInterest and other expense (income), net (Note 18)\n(26) \n71 \n146\nLoss on extinguishment of debt\n\n \n40 \n12\nIncome before income tax expense\n1,633\n \n1,877 \n1,151\nIncome tax expense\n130\n \n29 \n878\nNet income\n$\n1,503\n $\n1,848 $\n273\n \n \n \n \nEarnings per common share\n \n \n \n \nBasic\n$\n1.96\n $\n2.43 $\n0.36\nDiluted\n$\n1.95\n $\n2.40 $\n0.36\n \n \n \n \nWeighted-average number of shares outstanding\n \n \n \n \nBasic\n767\n \n762 \n754\nDiluted\n771\n \n771 \n766\nThe accompanying notes are an integral part of these Consolidated Financial Statements.\nF-5', 'doc_name': 'ACTIVISIONBLIZZARD_2019_10K', 'evidence_page_num': 69, 'evidence_text_full_page': 'Table of Contents\nACTIVISION BLIZZARD, INC. AND SUBSIDIARIES\nCONSOLIDATED STATEMENTS OF OPERATIONS\n(Amounts in millions, except per share data)\n \nFor the Years Ended December 31,\n \n2019\n \n2018\n \n2017\nNet revenues\n \n \n \n \nProduct sales\n$\n1,975\n $\n2,255 $\n2,110\nSubscription, licensing, and other revenues\n4,514\n \n5,245 \n4,907\nTotal net revenues\n6,489\n \n7,500 \n7,017\n \n \n \n \nCosts and expenses\n \n \n \n \nCost of revenuesproduct sales:\n \n \n \nProduct costs\n656\n \n719 \n733\nSoftware royalties, amortization, and intellectual property licenses\n240\n \n371 \n300\nCost of revenuessubscription, licensing, and other revenues:\n \n \n \nGame operations and distribution costs\n965\n \n1,028 \n984\nSoftware royalties, amortization, and intellectual property licenses\n233\n \n399 \n484\nProduct development\n998\n \n1,101 \n1,069\nSales and marketing\n926\n \n1,062 \n1,378\nGeneral and administrative\n732\n \n822 \n745\nRestructuring and related costs\n132\n \n10 \n15\nTotal costs and expenses\n4,882\n \n5,512 \n5,708\n \n \n \n \nOperating income\n1,607\n \n1,988 \n1,309\nInterest and other expense (income), net (Note 18)\n(26) \n71 \n146\nLoss on extinguishment of debt\n\n \n40 \n12\nIncome before income tax expense\n1,633\n \n1,877 \n1,151\nIncome tax expense\n130\n \n29 \n878\nNet income\n$\n1,503\n $\n1,848 $\n273\n \n \n \n \nEarnings per common share\n \n \n \n \nBasic\n$\n1.96\n $\n2.43 $\n0.36\nDiluted\n$\n1.95\n $\n2.40 $\n0.36\n \n \n \n \nWeighted-average number of shares outstanding\n \n \n \n \nBasic\n767\n \n762 \n754\nDiluted\n771\n \n771 \n766\nThe accompanying notes are an integral part of these Consolidated Financial Statements.\nF-5\n'}]"

    user_prompt = f"""
        Example of a 'simple retrieval' question:
        Question: What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.
        Answer: $1,577.00
        Evidence: {simple_retrieval_evidence}

        Example of a 'reasoning' question:
        Question: What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.
        Answer: 24.26
        Evidence: {reasoning_evidence}

        ------------------------------------------------------------
        Now, classify the following question:
        Question: {question}
        Answer: {answer}
        Evidence: {evidence}
    """
    response = openai.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=QuestionAudit
    )
    return response.choices[0].message.parsed
            

def run_audit(openai: OpenAI, df: DataFrame) -> DataFrame:
    nan_rows = df['question_reasoning'].isna()
    rows_to_audit = df[nan_rows]
    print(f"Total rows to audit: {len(rows_to_audit)}")

    for index, row in rows_to_audit.iterrows():
        question = row["question"]
        answer = row["answer"]
        evidence = row["evidence"]
        audit = audit_question(openai, question, answer, evidence)
        df.at[index, "audit_label"] = audit.label
        df.at[index, "audit_explanation"] = audit.reasoning
        print(f"Finished auditing {row["financebench_id"]}")

    return df

def remove_simple_retrieval(df: DataFrame, output_path: str):
    filtered_df = df[df["audit_label"] != "simple_retrieval"].drop(columns=["audit_label", "audit_explanation"])
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered dataset saved to {output_path}")

if __name__ == "__main__":
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    else:
        print("OPENAI_API_KEY is set")

    openai = OpenAI(api_key=openai_api_key)
    input_path = "data/filtered/financebench_filtered.csv"
    output_dir = "data/audited"
    audit_output_path = os.path.join(output_dir, "financebench_audited.csv")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(input_path)

    if not os.path.exists(audit_output_path):
        audited_df = run_audit(openai, df)
        audited_df.to_csv(audit_output_path, index=False)
        print(f"Dataset saved to {audit_output_path}")
    else:
        audited_df = pd.read_csv(audit_output_path)
    
    filtered_output_path = os.path.join(output_dir, "financebench_filtered_reasoning.csv")
    simple_retrieval_df = remove_simple_retrieval(audited_df, filtered_output_path)