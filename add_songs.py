import tiktoken
import os
import pinecone
import constants
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.vectorstores import Pinecone
from lyricsgenius import Genius
from langchain.text_splitter import RecursiveCharacterTextSplitter


os.environ['OPENAI_API_KEY'] = constants.OPENAI_API_KEY

def load_pinecone(docs):
    pinecone.init(
    api_key=constants.PINECONE_API_KEY,
    environment=constants.PINECONE_ENV
    )

    if constants.PINECONE_INDEX_NAME not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=constants.PINECONE_INDEX_NAME,
            dimension=1536 
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")    

    return Pinecone.from_documents(docs, embeddings, index_name=constants.PINECONE_INDEX_NAME)

def songs_to_add():
    songs = []

    with open('songs.txt') as f:
        songs = f.read().splitlines()

    return songs    

def get_text_chunks_metadata(texts,songs,chunk_size=400):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    total_chunks = []

    for id,text in enumerate(texts):
        chunks = text_splitter.split_documents(text)
        for idx,chunk in enumerate(chunks):
            chunks[idx].metadata['spotify_link'] = songs[id]
            chunks[idx].metadata['youtube_link'] = songs[id]

        total_chunks = total_chunks + chunks

    docs = [Document(page_content=c) for c in total_chunks]

    return docs    

    
def get_songs(songs):
    genius = Genius(constants.GENIUS_CLIENT_ACCESS_TOKEN)
    lyrics = []
    artist = []

    for song in songs:
        song_artist = song.split(' ')

        artist.append(song_artist)

        artist = genius.search_artist(song_artist[0], max_songs=3, sort="title")

        song = genius.search_song(song_artist[1], artist.name)

        lyrics.append(song.lyrics)

    return lyrics, artist    

def add_songs():
    songs = []
    with open('songs.txt') as f:
        songs = f.read().splitlines()

    lyrics, artist = get_songs(songs)

    docs =  get_text_chunks_metadata(lyrics,artist,500)

    index  = load_pinecone(docs)  

def main():
    add_songs()

if __name__ == "__main__":
    # calling the main function
    main()
