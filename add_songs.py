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


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_text_chunks_metadata(texts,songs,chunk_size=400):
    docs = [Document(page_content=t, metadata={"spotify_link":songs[i],"youtube_link":songs[i]}) for i,t in enumerate(texts)]

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=20,  # number of tokens overlap between chunks
    )
  
    chunks = text_splitter.split_documents(docs)

    return chunks   

    
def get_songs(songs):
    genius = Genius(constants.GENIUS_CLIENT_ACCESS_TOKEN)
    lyrics = []
    musician = []

    for song in songs:
        song_artist = song.split(',')

        print(f"Song: {song_artist[1]} by {song_artist[0]} is being added")

        musician.append(song_artist[0])

        artist = genius.search_artist(song_artist[0], max_songs=1, sort="title")

        song = genius.search_song(song_artist[1], artist.name)

        lyrics.append(song.lyrics)

        print(f"Song: {song_artist[1]} by {song_artist[0]} added")

    return lyrics, musician    

def add_songs():
    songs = []
    with open('songs.txt') as f:
        songs = f.read().splitlines()

    lyrics, artist = get_songs(songs)

    docs =  get_text_chunks_metadata(lyrics,artist,800)

    index  = load_pinecone(docs)  

def main():
    add_songs()

if __name__ == "__main__":
    # calling the main function
    main()
