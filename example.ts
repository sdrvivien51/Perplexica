import { SearchAgent } from './search';
import { ChatOpenAI } from '@langchain/openai';
import { OpenAIEmbeddings } from '@langchain/openai';

// Example usage
async function main() {
  // Initialize models
  const llm = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    openAIApiKey: process.env.OPENAI_API_KEY
  });

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY
  });

  // Create search agent
  const searchAgent = new SearchAgent({
    similarityMeasure: 'cosine',
    rerankThreshold: 0.3,
    maxResults: 10
  });

  // Perform search
  const query = "What is quantum computing?";
  const answer = await searchAgent.search(query, llm, embeddings);
  
  console.log('Answer:', answer);
}

main().catch(console.error);