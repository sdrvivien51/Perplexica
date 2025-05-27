import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { Embeddings } from '@langchain/core/embeddings';
import { BaseMessage } from '@langchain/core/messages';
import { Document } from '@langchain/core/documents';
import { StringOutputParser } from '@langchain/core/output_parsers';
import {
  RunnableLambda,
  RunnableMap,
  RunnableSequence,
} from '@langchain/core/runnables';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
} from '@langchain/core/prompts';
import axios from 'axios';
import { htmlToText } from 'html-to-text';
import { RecursiveCharacterTextSplitter } from '@langchain/text_splitters';
import computeDot from 'compute-dot';
import cosineSimilarity from 'compute-cosine-similarity';

interface SearchConfig {
  similarityMeasure?: 'cosine' | 'dot';
  rerankThreshold?: number;
  maxResults?: number;
}

const defaultConfig: SearchConfig = {
  similarityMeasure: 'cosine',
  rerankThreshold: 0.3,
  maxResults: 15
};

export class SearchAgent {
  private config: SearchConfig;
  private strParser = new StringOutputParser();

  constructor(config: Partial<SearchConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  private computeSimilarity(x: number[], y: number[]): number {
    if (this.config.similarityMeasure === 'cosine') {
      return cosineSimilarity(x, y) as number;
    } else {
      return computeDot(x, y);
    }
  }

  private async searchWeb(query: string) {
    try {
      const url = new URL(`${process.env.SEARXNG_URL}/search`);
      url.searchParams.append('q', query);
      url.searchParams.append('format', 'json');
      url.searchParams.append('language', 'en');

      const response = await axios.get(url.toString());
      return response.data.results;
    } catch (error) {
      console.error('Search error:', error);
      return [];
    }
  }

  private async getDocumentsFromUrls(urls: string[]): Promise<Document[]> {
    const splitter = new RecursiveCharacterTextSplitter();
    const docs: Document[] = [];

    await Promise.all(
      urls.map(async (url) => {
        try {
          const response = await axios.get(url);
          const text = htmlToText(response.data, {
            selectors: [{ selector: 'a', options: { ignoreHref: true } }]
          })
            .replace(/(\r\n|\n|\r)/gm, ' ')
            .replace(/\s+/g, ' ')
            .trim();

          const chunks = await splitter.splitText(text);
          const title = response.data.match(/<title.*>(.*?)<\/title>/)?.[1] || url;

          docs.push(
            ...chunks.map(
              chunk =>
                new Document({
                  pageContent: chunk,
                  metadata: { title, url }
                })
            )
          );
        } catch (error) {
          console.error(`Error processing ${url}:`, error);
        }
      })
    );

    return docs;
  }

  private async rerankDocs(
    query: string,
    docs: Document[],
    embeddings: Embeddings
  ): Promise<Document[]> {
    if (docs.length === 0) return [];

    const [queryEmbedding, docEmbeddings] = await Promise.all([
      embeddings.embedQuery(query),
      embeddings.embedDocuments(docs.map(doc => doc.pageContent))
    ]);

    const similarities = docEmbeddings.map((docEmbedding, i) => ({
      index: i,
      similarity: this.computeSimilarity(queryEmbedding, docEmbedding)
    }));

    return similarities
      .filter(sim => sim.similarity > (this.config.rerankThreshold || 0.3))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, this.config.maxResults)
      .map(sim => docs[sim.index]);
  }

  public async search(
    query: string,
    llm: BaseChatModel,
    embeddings: Embeddings,
    history: BaseMessage[] = []
  ) {
    // Search chain
    const searchChain = RunnableSequence.from([
      PromptTemplate.fromTemplate(
        `Rephrase the following query for web search, making it more focused and specific:
        Query: {query}
        Rephrased query:`
      ),
      llm,
      this.strParser,
      RunnableLambda.from(async (searchQuery: string) => {
        const results = await this.searchWeb(searchQuery);
        const docs = await this.getDocumentsFromUrls(
          results.map((r: any) => r.url)
        );
        return this.rerankDocs(searchQuery, docs, embeddings);
      })
    ]);

    // Answer chain
    const answerChain = RunnableSequence.from([
      RunnableMap.from({
        question: (input: any) => input.query,
        context: (input: any) => 
          input.docs
            .map((doc: Document, i: number) => 
              `[${i + 1}] ${doc.metadata.title}\n${doc.pageContent}`
            )
            .join('\n\n'),
        chat_history: () => history
      }),
      ChatPromptTemplate.fromMessages([
        ['system', `You are an AI assistant that helps users by providing detailed, accurate answers based on the given context. 
        Always cite your sources using [number] notation.`],
        new MessagesPlaceholder('chat_history'),
        ['user', '{question}'],
        ['system', 'Context:\n{context}']
      ]),
      llm,
      this.strParser
    ]);

    // Execute search and answer
    const docs = await searchChain.invoke({ query });
    return answerChain.invoke({ query, docs });
  }
}