from VectorStore import DocsToVectorStore

minutes_folder = 'Minutes/'
vs = DocsToVectorStore(minutes_folder)
vs.load_docs()

# --Add year, month, and day to METADATA--
for doc in vs.docs:
    datetimestr = doc.metadata['source'][-12:-4]
    doc.metadata['year'] = int(datetimestr[:4])
    doc.metadata['month'] = int(datetimestr[4:6])
    doc.metadata['day'] = int(datetimestr[6:])

# --Split documents into sentences--
vs.split_docs()

# --Embed sentences into vectors--
vs.embed_docs()
vs.save_docstore('MINUTES_FOMC_HISTORY')
