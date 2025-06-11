from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from fuzzywuzzy import process
from transformers import pipeline

companyDict = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN"
}

FUZZY_THRESHOLD = 80
ZEROSHOT_THRESHOLD = 0.5
zeroShotClassificationModelName = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=zeroShotClassificationModelName)

def extractTicker(query, companyDict, topN=3):
    fuzzyMatch = process.extractOne(query, companyDict.keys(), limit=topN)
    fuzzyTickers = [companyDict[name] for name, score in fuzzyMatch if score >= FUZZY_THRESHOLD]
    
    if len(fuzzyTickers) < topN:
        candidateLabels = list(companyDict.keys())
        result = classifier(query, candidateLabels)
        zslTickers = [
            companyDict[label] for label, score in zip(result['labels'], result['scores'])
            if score >= ZEROSHOT_THRESHOLD
        ]        
        
        allTickers = list(dict.fromkeys(fuzzyTickers + zslTickers))
        return allTickers[:topN]

    return fuzzyTickers[:topN]
    

modelName = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForSeq2SeqLM.from_pretrained(modelName)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

promptTemplate = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    You are a financial assistant. Based on the stock data below, answer the question.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer in a helpful and concise manner.
    """
)

ragChain = LLMChain(llm=llm, prompt=promptTemplate)
