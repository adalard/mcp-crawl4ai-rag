"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, BinaryIO, Union
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import openai
import re
import time
import tempfile
import requests
from pathlib import Path
import PyPDF2
import io

# Load OpenAI API key for embeddings
llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
if llm_provider == "openai":
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Ollama base URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# LLM Provider Abstraction Layer
class LLMResponse:
    """Class to standardize responses from different LLM providers"""
    def __init__(self, content: str, model: str):
        self.content = content
        self.model = model

def get_llm_provider() -> str:
    """Get the configured LLM provider"""
    return os.getenv("LLM_PROVIDER", "openai").lower()

def get_embedding_model() -> str:
    """Get the embedding model based on the LLM provider"""
    if get_llm_provider() == "ollama":
        return os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    else:
        return "text-embedding-3-small"  # Default OpenAI embedding model

def create_chat_completion(model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> LLMResponse:
    """Create a chat completion using the configured LLM provider
    
    Args:
        model: Model name to use (interpreted based on LLM provider)
        messages: List of message dictionaries with role and content
        temperature: Temperature for the model (0.0 to 1.0)
        
    Returns:
        LLMResponse object with generated content
    """
    provider = get_llm_provider()
    
    if provider == "openai":
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return LLMResponse(response.choices[0].message.content, model)
        except Exception as e:
            print(f"Error creating OpenAI chat completion: {e}")
            return LLMResponse("Error generating response", model)
    
    elif provider == "ollama":
        try:
            # Format messages for Ollama API
            ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            # Make API call to Ollama
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "options": {"temperature": temperature},
                    "stream": False  # Explicitly disable streaming to get a complete response
                }
            )
            
            if response.status_code == 200:
                # Handle potential streaming/multi-line response
                try:
                    result = response.json()
                    return LLMResponse(result["message"]["content"], model)
                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error: {json_err}")
                    # Try to parse the first valid JSON object from the response
                    response_text = response.text.strip()
                    try:
                        # Find the first complete JSON object
                        first_json_end = response_text.find("}\n")
                        if first_json_end > 0:
                            first_json = response_text[:first_json_end + 1]
                            result = json.loads(first_json)
                            return LLMResponse(result["message"]["content"], model)
                    except Exception as parse_err:
                        print(f"Failed to parse streaming response: {parse_err}")
                    
                    # Fallback to returning the raw text if JSON parsing fails
                    return LLMResponse(f"Error parsing response: {response_text[:100]}...", model)
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return LLMResponse(f"Error: {response.status_code}", model)
        except Exception as e:
            print(f"Error creating Ollama chat completion: {e}")
            return LLMResponse("Error generating response", model)
    
    else:
        print(f"Unsupported LLM provider: {provider}")
        return LLMResponse("Unsupported LLM provider", model)

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")
    
    return create_client(url, key)

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    provider = get_llm_provider()
    embedding_model = get_embedding_model()
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    if provider == "openai":
        # OpenAI embedding logic
        for retry in range(max_retries):
            try:
                response = openai.embeddings.create(
                    model=embedding_model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                    # Try creating embeddings one by one as fallback
                    print("Attempting to create embeddings individually...")
                    embeddings = []
                    successful_count = 0
                    
                    for i, text in enumerate(texts):
                        try:
                            individual_response = openai.embeddings.create(
                                model=embedding_model,
                                input=[text]
                            )
                            embeddings.append(individual_response.data[0].embedding)
                            successful_count += 1
                        except Exception as individual_error:
                            print(f"Failed to create embedding for text {i}: {individual_error}")
                            # Add zero embedding as fallback
                            embeddings.append([0.0] * 1536)
                    
                    print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                    return embeddings
    
    elif provider == "ollama":
        # Ollama embedding logic
        embeddings = []
        successful_count = 0
        failed_indices = []
        
        # Process embeddings in batches for Ollama (since it doesn't support batch embedding natively)
        for i, text in enumerate(texts):
            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": embedding_model,
                        "input": text
                    }
                )
                
                if response.status_code == 200:
                    try:
                        embedding_data = response.json()
                        if "embedding" in embedding_data:
                            embeddings.append(embedding_data["embedding"])
                            successful_count += 1
                        else:
                            print(f"No embedding found in response for text {i}")
                            embeddings.append([0.0] * 1536)  # Default size for compatibility
                            failed_indices.append(i)
                    except json.JSONDecodeError as json_err:
                        print(f"JSON decode error for embedding {i}: {json_err}")
                        # Try to extract the embedding from potentially malformed JSON
                        try:
                            # Find the embedding array in the response text
                            response_text = response.text.strip()
                            # Look for the embedding array pattern
                            import re
                            embedding_match = re.search(r'"embedding":\s*\[(.*?)\]', response_text)
                            if embedding_match:
                                # Parse the embedding array
                                embedding_str = embedding_match.group(1)
                                embedding_values = [float(x) for x in embedding_str.split(',')]
                                embeddings.append(embedding_values)
                                successful_count += 1
                                print(f"Successfully extracted embedding from malformed JSON for text {i}")
                            else:
                                print(f"Could not extract embedding from response for text {i}")
                                embeddings.append([0.0] * 1536)  # Default size for compatibility
                                failed_indices.append(i)
                        except Exception as extract_err:
                            print(f"Failed to extract embedding from response: {extract_err}")
                            embeddings.append([0.0] * 1536)  # Default size for compatibility
                            failed_indices.append(i)
                else:
                    print(f"Error creating embedding for text {i}: {response.status_code} - {response.text}")
                    embeddings.append([0.0] * 1536)  # Default size for compatibility
                    failed_indices.append(i)
                    
                # Add a small delay to avoid rate limiting
                if i < len(texts) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Exception creating embedding for text {i}: {e}")
                embeddings.append([0.0] * 1536)  # Default size for compatibility
                failed_indices.append(i)
        
        if failed_indices:
            print(f"Failed to create embeddings for {len(failed_indices)}/{len(texts)} texts")
        
        print(f"Successfully created {successful_count}/{len(texts)} embeddings with Ollama")
        return embeddings
    
    else:
        print(f"Unsupported LLM provider: {provider}")
        # Return zero embeddings as fallback
        return [[0.0] * 1536 for _ in texts]

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using the configured LLM provider.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    # Check if we should use contextual embeddings
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false").lower() == "true"
    
    if not use_contextual_embeddings or not full_document:
        return chunk, False
    
    model_choice = os.getenv("MODEL_CHOICE")
    
    if not model_choice:
        return chunk, False
    
    try:
        # Prepare the prompt for generating contextual information
        system_prompt = """
        You are an expert at understanding document context and summarization.
        Your task is to provide contextual information about how a specific chunk of text fits 
        within the larger document.
        
        Focus on:
        1. Where this chunk appears in the document structure
        2. What topics this chunk relates to
        3. How this chunk relates to the document's main themes
        4. Any key entities, concepts, or terminology that help situate this chunk
        
        Keep your response brief (under 100 words).
        """
        
        # Call the LLM using our abstraction layer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Full document:\n\n{full_document[:10000]}\n\nChunk to contextualize:\n\n{chunk}"}
        ]
        
        llm_response = create_chat_completion(
            model=model_choice,
            messages=messages,
            temperature=0.5
        )
        
        contextual_info = llm_response.content.strip()
        
        # Combine original chunk with contextual information
        combined_text = f"{contextual_info}\n\n{chunk}"
        return combined_text, True
        
    except Exception as e:
        print(f"Error generating contextual embedding: {e}")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

def add_documents_to_supabase(
    client: Client, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Delete existing records for these URLs in a single operation
    try:
        if unique_urls:
            # Use the .in_() filter to delete all records with matching URLs
            client.table("crawled_pages").delete().in_("url", unique_urls).execute()
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))
            
            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                for idx, arg in enumerate(process_args)}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])
            
            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)
        
        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])
            
            # Extract source_id from URL
            parsed_url = urlparse(batch_urls[j])
            source_id = parsed_url.netloc or parsed_url.path
            
            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                "source_id": source_id,  # Add source_id field
                "embedding": batch_embeddings[j]  # Use embedding from contextual content
            }
            
            batch_data.append(data)
        
        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                client.table("crawled_pages").insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Supabase (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table("crawled_pages").insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {record['url']}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")

def search_documents(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    # Execute the search using the match_crawled_pages function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params['filter'] = filter_metadata  # Pass the dictionary directly, not JSON-encoded
        
        result = client.rpc('match_crawled_pages', params).execute()
        
        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks


def generate_code_example_summary(code: str, context_before: str, context_after: str):
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Default summary if something goes wrong
    default_summary = "Code example"
    
    if not model_choice:
        return default_summary
    
    try:
        # Create a prompt to generate the summary
        prompt = f"""You are a technical writer specializing in code documentation. Please provide a concise one-sentence summary that describes what this code example demonstrates, based on the code itself and its surrounding context.

Context before the code:
{context_before[:1000]}

Code example:
```
{code[:2000]}
```

Context after the code:
{context_after[:1000]}

Write ONLY the summary, with no additional explanation, prefix, or quotes. The summary should be a single sentence with 15 words or less. Focus on the specific functionality or technique being demonstrated."""

        # Call the LLM using our abstraction layer
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in technical documentation."},
            {"role": "user", "content": prompt}
        ]
        
        llm_response = create_chat_completion(
            model=model_choice,
            messages=messages,
            temperature=0.3
        )
        
        summary = llm_response.content.strip()
        
        # Remove any quotes that might be in the response
        summary = summary.strip('"').strip("'").strip()
        
        # If the summary is too long, truncate it
        if len(summary) > 100:
            summary = summary[:97] + "..."
        
        return summary
        
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return default_summary


def add_code_examples_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
):
    """
    Add code examples to the Supabase code_examples table in batches.
    
    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return
        
    # Delete existing records for these URLs
    unique_urls = list(set(urls))
    for url in unique_urls:
        try:
            client.table('code_examples').delete().eq('url', url).execute()
        except Exception as e:
            print(f"Error deleting existing code examples for {url}: {e}")
    
    # Process in batches
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = []
        
        # Create combined texts for embedding (code + summary)
        for j in range(i, batch_end):
            combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
            batch_texts.append(combined_text)
        
        # Create embeddings for the batch
        embeddings = create_embeddings_batch(batch_texts)
        
        # Check if embeddings are valid (not all zeros)
        valid_embeddings = []
        for embedding in embeddings:
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
            else:
                print(f"Warning: Zero or invalid embedding detected, creating new one...")
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(batch_texts[len(valid_embeddings)])
                valid_embeddings.append(single_embedding)
        
        # Prepare batch data
        batch_data = []
        for j, embedding in enumerate(valid_embeddings):
            idx = i + j
            
            # Extract source_id from URL
            parsed_url = urlparse(urls[idx])
            source_id = parsed_url.netloc or parsed_url.path
            
            batch_data.append({
                'url': urls[idx],
                'chunk_number': chunk_numbers[idx],
                'content': code_examples[idx],
                'summary': summaries[idx],
                'metadata': metadatas[idx],  # Store as JSON object, not string
                'source_id': source_id,
                'embedding': embedding
            })
        
        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                client.table('code_examples').insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into Supabase (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table('code_examples').insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record for URL {record['url']}: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")
        print(f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples")


def update_source_info(client: Client, source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources table.
    
    Args:
        client: Supabase client
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        # Try to update existing source
        result = client.table('sources').update({
            'summary': summary,
            'total_word_count': word_count,
            'updated_at': 'now()'
        }).eq('source_id', source_id).execute()
        
        # If no rows were updated, insert new source
        if not result.data:
            client.table('sources').insert({
                'source_id': source_id,
                'summary': summary,
                'total_word_count': word_count
            }).execute()
            print(f"Created new source: {source_id}")
        else:
            print(f"Updated source: {source_id}")
            
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using the configured LLM provider.
    
    This function uses the configured LLM to generate a concise summary of the source content.
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")
    
    if not model_choice:
        return f"Content from {source_id}"
    
    try:
        # Prepare a sample of the content
        # Take the first part of the content
        sample = content[:20000]  # Use first 20k characters as a sample
        
        # Create the prompt for the summary generation
        prompt = f"""Please provide a concise summary of the content from the website {source_id}. Focus on the main topics, themes, and the type of information provided. Keep the summary under {max_length} characters.

Content sample:
{sample}

Summary:"""
        
        # Call the LLM using our abstraction layer
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides concise summaries of web content."},
            {"role": "user", "content": prompt}
        ]
        
        llm_response = create_chat_completion(
            model=model_choice,
            messages=messages,
            temperature=0.7
        )
        
        # Extract the summary from the response
        summary = llm_response.content.strip()
        
        # Truncate if necessary
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."
            
        return summary
        
    except Exception as e:
        print(f"Error extracting summary for {source_id}: {e}")
        return f"Content from {source_id}"


def search_code_examples(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results
        
    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    # Since code examples are embedded with their summaries, we should make the query more descriptive
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)
    
    # Execute the search using the match_code_examples function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params['filter'] = filter_metadata
            
        # Add source filter if provided
        if source_id:
            params['source_filter'] = source_id
        
        result = client.rpc('match_code_examples', params).execute()
        
        return result.data
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []


def is_pdf(url: str) -> bool:
    """
    Check if a URL is a PDF file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a PDF file, False otherwise
    """
    return url.lower().endswith('.pdf')


def download_pdf(url: str) -> Optional[bytes]:
    """
    Download a PDF file from a URL.
    
    Args:
        url: URL of the PDF file
        
    Returns:
        PDF file content as bytes, or None if download fails
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Check if the content type is PDF
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
            print(f"Warning: URL {url} does not appear to be a PDF (Content-Type: {content_type})")
        
        return response.content
    except Exception as e:
        print(f"Error downloading PDF from {url}: {e}")
        return None


def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_content: PDF file content as bytes
        
    Returns:
        Extracted text as a string
    """
    try:
        # Create a PDF reader object
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from each page
        text = []
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text.append(page_text)
            except Exception as page_error:
                print(f"Error extracting text from page {page_num}: {page_error}")
        
        # Join all pages with double newlines for better separation
        extracted_text = "\n\n".join(text)
        print(f"Successfully extracted {len(text)} pages from PDF, total length: {len(extracted_text)} characters")
        return extracted_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def process_pdf_url(url: str) -> Optional[Dict[str, str]]:
    """
    Process a PDF URL: download the PDF and extract its text.
    
    Args:
        url: URL of the PDF file
        
    Returns:
        Dictionary with URL and extracted text, or None if processing fails
    """
    try:
        # Download the PDF
        pdf_content = download_pdf(url)
        if not pdf_content:
            return None
        
        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_content)
        if not text:
            return None
        
        # Return the result as a dictionary
        return {'url': url, 'markdown': text}
    except Exception as e:
        print(f"Error processing PDF URL {url}: {e}")
        return None


def process_pdf_batch(urls: List[str], max_workers: int = 5) -> List[Dict[str, str]]:
    """
    Process multiple PDF URLs in parallel.
    
    Args:
        urls: List of PDF URLs to process
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of dictionaries with URL and extracted text
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(process_pdf_url, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing PDF URL {url}: {e}")
    
    return results


def smart_chunk_markdown(text: str, chunk_size: int = 2500) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs.
    
    For PDFs and other complex documents, we use a smaller default chunk size (2500)
    to ensure better compatibility with embedding models.
    """
    # Check if text is empty or None
    if not text:
        print("Warning: Empty text provided to smart_chunk_markdown")
        return []
        
    # For PDFs, we'll use a more aggressive chunking strategy
    is_pdf_content = '\f' in text  # Form feed character often appears in PDFs
    
    # Use smaller chunks for PDFs to avoid embedding issues
    if is_pdf_content:
        print(f"Detected PDF content, using smaller chunk size for better embedding compatibility")
        # Use a smaller chunk size for PDFs to avoid embedding issues
        chunk_size = min(chunk_size, 2500)
    
    chunks = []
    start = 0
    text_length = len(text)
    
    print(f"Chunking text of length {text_length} with chunk size {chunk_size}")

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        
        # If we found a code block marker and it's not at the very beginning
        if code_block > 0 and code_block > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
            end = start + code_block

        # If no code block boundary, try to find a paragraph break
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    print(f"Created {len(chunks)} chunks from text")
    return chunks


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }


def process_code_example(args):
    """
    Process a single code example to generate its summary.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (code, context_before, context_after)
        
    Returns:
        The generated summary
    """
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)