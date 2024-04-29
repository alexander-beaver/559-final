# pip install accelerate
import gzip
import os
import shutil
import urllib
import openai
import os
import bz2
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

#from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

'''
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
'''
# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

cols = "INDX, TS, TYP, KERNEL, INC, PHYSIN, OUTC, PHYSOUT, SRC, DST, LEN, TOS, PREC, TTL, ID, PROTO, SPT, DPT, WNDW, RES, SYN, URGP."


class Assistant:
    """Gemma 2b based assistant that replies given the retrieved documents"""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        self.Gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", offload_buffers=True)

    def create_prompt(self, query, retrieved_info):
        # instruction to areply to query given the retrived information
        prompt = f"""You are a security analyst looking through RedHat, IPTables, and SNORT IDS logs. You are looking for potential security threats. Idenfiy the potential security threats in the logs. Be detailed, use simple words and examples in your explanations. If required, utilize the relevant information.
        Logs: {query}
        Relevant information: {retrieved_info}
        Output:
        """
        return prompt

    def reply(self, query, retrieved_info):
        prompt = self.create_prompt(query, retrieved_info)
        input_ids = self.tokenizer(query, return_tensors="pt", max_length=len(prompt)+20).input_ids.to('cuda')
        # Generate text with a focus on factual responses
        generated_text = self.Gemma.generate(
            input_ids,
            max_length=len(prompt)+100,  # let answers be not that long
            temperature=0.7,  # Adjust temperature according to the task, for code generation it can be 0.9
        )
        # Decode and return the answer
        answer = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)
        return answer

import os
def get_all_pdfs(directory):
    """Get the list of pdf files in the directory."""
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


class Retriever:
    """Sentence embedding based Retrieval Based Augmented generation.
        Given database of pdf files, retriever finds num_retrieved_docs relevant documents"""
    def __init__(self, num_retrieved_docs=1, pdf_folder_path='./context'):
        # load documents
        print("Loading documents")
        pdf_files = get_all_pdfs(pdf_folder_path)
        print("Documents used", pdf_files)
        log_files = [f for f in os.listdir() if f.endswith(".log")]
        print("Log files", log_files)
        loaders = [PyPDFLoader(pdf_file) for pdf_file in pdf_files]
        logloader = [TextLoader(log_file) for log_file in log_files]
        all_documents = []

        for loader in loaders:
            print("Loading", loader)
            raw_documents = loader.load()
            # split the documents into smaller chunks
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
            )
            documents = text_splitter.split_documents(raw_documents)
            all_documents.extend(documents)
        for loader in logloader:
            print("Loading", loader)
            raw_documents = loader.load()
            # split the documents into smaller chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
            )
            documents = text_splitter.split_documents(raw_documents)
            all_documents.extend(documents)
        '''
        for loader in logloader:
            print("Loading", loader)
            raw_documents = loader.load()
            # split the documents into smaller chunks
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
            )
            documents = text_splitter.split_documents(raw_documents)
            all_documents.extend(documents)
        '''
        # create a vectorstore database

        print("Creating vectorstore")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # fast model with competitive perfomance
        print("Embeddings created")
        self.db = FAISS.from_documents(all_documents, embeddings)
        print("Database created")
        self.retriever = self.db.as_retriever(search_kwargs={"k": len(all_documents)})
        print("Retriever initialized")
    def search(self, query):
        # retrieve top k similar documents to query
        docs = self.retriever.get_relevant_documents(query)
        return docs
chatbot = Assistant()
retriever = Retriever()
def execute_codegen_ai(prompt):
    # Construct the instruction-based prompt
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPEN_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

def execute_rag(prompt):
    global chatbot
    global retriever

    # Retrieve relevant documents
    retrieved_info = retriever.search(prompt)
    # Generate response
    response = chatbot.reply(prompt, retrieved_info)
    return response

def execute_llm(prompt):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", offload_buffers=True)
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=len(prompt)+20).to("cuda")
    print("Running model")

    outputs = model.generate(**input_ids, max_length=len(prompt)+20)
    out = tokenizer.decode(outputs[0])
    print(out)
    return out

def process_all_logs():
    logfiles = [f for f in os.listdir() if f.endswith(".log")]
    '''
    for logfile in logfiles:
        print("Processing", logfile)
        # Open file, split into 20 line chunks, and process each chunk
        with open(logfile, "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 10):
                print("Processing chunk", i)
                chunk = lines[i:i+20]
                joined_chunk = "\n".join(chunk)
                prompt = "I am a security analyst looking through these logs. Please give me a detailed description of anything that relate to 10.2.1.2. If no logs are identified, please state that.\n\n=====Logs======\n "+joined_chunk

                res = execute_rag(prompt)
                print(res)
                print("=====================================")
    '''
    for logfile in logfiles:
        print("Processing", logfile)
        # Open file, split into 20 line chunks, and process each chunk
        with open(logfile, "r") as f:
            lines = f.readlines()
            # Select lines that contain the IP address 10.2.1.2
            ip_lines = [line for line in lines if line.find("10.2.1.2") != -1]
            results = []
            # Chunk into 10 line chunks, and process each chunk
            for i in range(0, len(ip_lines), 5):
                print("Processing chunk", i)
                chunk = ip_lines[i:i+5]
                print("Chunk", chunk)
                joined_chunk = "\n".join(chunk)
                prompt = "I am a security analyst looking through these logs. Please give me a detailed description of the following logs. \n\n=====Logs======\n "+joined_chunk
                res = execute_rag(prompt)
                print(res)
                print("=====================================")
                results.append(res)
            #Summarize the results using RAG
            prompt = "I am a security analyst looking through these logs. Please summarize this summary in a way that is easy to understand.\n\n=====Summary======\n "+ "\n".join(results)
            res = execute_rag(prompt)
            print(res)
            print("DONE=====================================")

def generate_transformer_code():
    if not os.path.exists("transformer_code.py"):
        input_text = (
                    "You are writing a Python program that will take IPTables logs and parse them into a SQLite database. The program should be able to read the logs from a file and insert them into the database. The logs are in the following format:\n"
                    "Feb  1 00:00:02 bridge kernel: INBOUND TCP: IN=br0 PHYSIN=eth0 OUT=br0 PHYSOUT=eth1 SRC=192.150.249.87 DST=11.11.11.84 LEN=40 TOS=0x00 PREC=0x00 TTL=110 ID=12973 PROTO=TCP SPT=220 DPT=6129 WINDOW=16384 RES=0x00 SYN URGP=0 \n" +
                    "Write a program that will take the file at ./SotM30-anton.log and save it to log.db. The table should be called logs. The columns should be " + cols+". Then, read through each line in SotM30-anton.log and insert the data into the logs table. Double check for runtime errors before generating. If a value is not defined for a column, insert NULL.")
        out = execute_codegen_ai(input_text)
        print(out)
        out = "\n".join(out.split('```')[1].split("\n")[1:])

        outfile = open("transformer_code.py", "w")
        outfile.write(out)
        outfile.close()
        return out
    else:
        loaded = open("transformer_code.py", "r")
        out = loaded.read()
        loaded.close()
        return out



def query_log_db():
    import sqlite3
    conn = sqlite3.connect('log.db')
    c = conn.cursor()
    instruction_subset = execute_llm(
        "I am a security analyst looking through IPTables logs. Give me specific examples of things that I should look for in order to help me identify potential security threats.")
    input_text = (
                "You are provided a SQLite database with IPTables logs. You are a security analyst looking through the logs to identify potential security threats. Write a query that will return the logs that match the following criteria:\n" +
                instruction_subset +
                "\n\n The table is called logs. The columns are " + cols)
    sqlinst = execute_codegen_ai(input_text)
    print(sqlinst)
    c.execute(sqlinst)
    rows = c.fetchall()
    print(rows)


def run_query():
    process_all_logs()
    '''
    if not os.path.exists("transformer_code.py"):

        instruction_subset = execute_llm(
            "I am a security analyst looking through IPTables logs. Give me specific examples of things that I should look for in order to help me identify potential security threats.")
        input_text = (
                "There is a file called SotM30-anton.log which has IPTables logs. You are a security analyst looking through the logs to identify potential security threats. Write a query that will return the logs that match the following criteria:\n" +
                instruction_subset +
                "\n\n Write a python script that will open the file, read through it, and print the concerning output to standard output" + cols)
        out = execute_codegen_ai(input_text)
        print(out)
        out = "\n".join(out.split('```')[1].split("\n")[1:])
        print(out)
        outfile = open("transformer_code.py", "w")
        outfile.write(out)
        outfile.close()
        exec(out)
    else:
        loaded = open("transformer_code.py", "r")
        out = loaded.read()
        loaded.close()
        exec(out)
    '''




if __name__ == "__main__":
    #openai.api_key = os.getenv("OPEN_API_KEY")
    # If SotM30-anton.log does not exist, download it and uncompress using gzip
    if not os.path.exists("Bastion.tar"):
        print("Downloading file")
        url = "http://log-sharing.dreamhosters.com/Bastion.tar"
        urllib.request.urlretrieve(url, "Bastion.tar")
        print("Downloaded file")
        #extract the tar file
        import tarfile
        tar = tarfile.open("Bastion.tar")
        tar.extractall()
        tar.close()
        #There are a number of bz2 files in the tar file. Find how many there are, and decompress them
        import bz2
        import os
        import shutil
        import glob
        files = glob.glob("*.bz2")

        print("Files to decompress: ", files)
        for f in files:
            with open(f.replace(".bz2", ".log"), 'wb') as new_file, bz2.BZ2File(f, 'rb') as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)
            os.remove(f)



    print("File downloaded")




    '''
    if not os.path.exists("SotM30-anton.log"):
        url = "http://log-sharing.dreamhosters.com/SotM30-anton.log.gz"
        urllib.request.urlretrieve(url, "SotM30-anton.log.gz")
        with gzip.open("SotM30-anton.log.gz", "rb") as f_in:
            with open("SotM30-anton.log", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove("SotM30-anton.log.gz")
    '''

    run_query()
