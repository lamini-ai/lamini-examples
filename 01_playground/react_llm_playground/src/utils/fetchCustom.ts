export async function fetchStreamingCustom(
  question: string,
  baseModel: string | null,
  setModelResponse: any,
  token: string,
  output_len: number
) {
  var myHeaders = new Headers();
  //myHeaders.append("Authorization", "Bearer " + token);
  myHeaders.append("Content-Type", "application/json");
  // TODO: remove this hack so our finetuned model can also get data context

  let finalQuestion = question;
  if (
    baseModel === "meta-llama/Llama-2-7b-chat-hf" ||
    baseModel === "meta-llama/Llama-2-13b-chat-hf" ||
    baseModel === "meta-llama/Llama-2-70b-chat-hf"
  ) {
    finalQuestion =
      `<s>[INST] <<SYS>>
      You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

      If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
      <</SYS>>

      ` +
      question +
      ` [/INST]`;
  } else if (baseModel == "Intel/neural-chat-7b-v3-1") {
    finalQuestion =
      `### System:
    {Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.}
    ### User:
    ` +
      question +
      `
    ### Assistant:`;
  } else if (baseModel == "mistralai/Mistral-7B-Instruct-v0.1") {
    finalQuestion =
      `<s>[INST {input:task_description}` +
      question +
      ` [/INST]`;
  }
  const body = {
    //id: "website",
    model_name: baseModel,
    out_type: { answer: "string" },
    prompt: [finalQuestion],
    max_tokens: output_len,
  };

  // Define a polling interval in milliseconds
  const pollingInterval = 1; // polling interval in milliseconds
  const maxDurationInSeconds = 150; // maximum duration in seconds
  const giveUpAfter = 3; // give up after 3 seconds
  // Use a flag to control the polling loop
  let stopPolling = false;
  let hasReceivedFirstToken = false;
  let elapsedTime = 0;
  const startTime = Date.now();
  while (!stopPolling) {
    if (elapsedTime > giveUpAfter * 1000 && !hasReceivedFirstToken) {
      setModelResponse(
        "Oops, the llama is running late due to traffic. Please try again."
      );
      return;
    }
    if (elapsedTime > maxDurationInSeconds * 1000) {
      return;
    }
    ({ stopPolling, hasReceivedFirstToken } = await fetchStreamingCustomOne(
      body,
      myHeaders,
      setModelResponse,
      startTime,
      hasReceivedFirstToken
    ));

    // Wait for the next polling interval
    await new Promise((resolve) => setTimeout(resolve, pollingInterval));

    elapsedTime = Date.now() - startTime;
  }
}

export async function fetchStreamingCustomOne(
  body: any,
  myHeaders: any,
  setModelResponse: any,
  startTime: any,
  hasReceivedFirstToken: boolean
) {
  const maxTimeToFirstToken = 3; // maximum duration for wait to get first token in seconds
  const controller = new AbortController();
  const timeoutId = setTimeout(
    () => controller.abort(),
    maxTimeToFirstToken * 1000
  ); // 3000 milliseconds timeout
  try {
    const response = await fetch(
      process.env.REACT_APP_API_URL + "/streaming_generate",
      {
        method: "POST",
        body: JSON.stringify(body),
        headers: myHeaders,
        signal: controller.signal, // Pass the abort signal to the fetch options
      }
    );
    const responseJson = await response.json();
    clearTimeout(timeoutId); // Clear the timeout if the request is successful
    if (!response.ok) {
      if (responseJson.detail.detail) {
        setModelResponse(responseJson.detail.detail);
      } else {
        if (response.status === 513) {
          setModelResponse(
            "I am downloading the model. Please try again in a few minutes."
          );
        } else if (response.status === 503) {
          setModelResponse(
            "Servers are not available at this time. Please try again later."
          );
        } else if (response.status === 561) {
          setModelResponse(
            "Sorry, you don't have access to this model. Need help? Email us at info@lamini.ai"
          );
        } else if (response.status === 402) {
          setModelResponse(
            "Sorry, you don't have enough credits. Please upgrade to Pro in your account or contact us at info@lamini.ai"
          );
        } else {
          setModelResponse(
            "Oops, something went wrong. I couldn’t provide you with a response. Please try again."
          );
        }
      }
      return { stopPolling: true, hasReceivedFirstToken };
    }
    // Initialize an empty response
    let partialResponse = "";
    let stopPolling = responseJson.status[0];
    // Append the received chunk to the partial response
    partialResponse += responseJson.data[0].answer;
    // Update the model response in the state
    if (partialResponse === "") {
      if (Date.now() - startTime > maxTimeToFirstToken * 1000) {
        setModelResponse(
          "Oops, the llama is running late due to traffic. Please try again."
        );
        return { stopPolling: true, hasReceivedFirstToken };
      }
      // Don't set the ModelResponse else it would remove the loading dots
      return { stopPolling: false, hasReceivedFirstToken };
    }
    setModelResponse(partialResponse);
    return { stopPolling, hasReceivedFirstToken: true };
  } catch (error) {
    // Handle abort error
    if (error.name === "AbortError") {
      return { stopPolling: false, hasReceivedFirstToken };
    } else if (
      error instanceof TypeError &&
      error.message === "Failed to fetch"
    ) {
      // Handle network errors
      setModelResponse(
        "Oops, a network error occurred. Please try again in a few minutes."
      );
    } else {
      // Handle other errors
      setModelResponse("Oops, something went wrong. Please try again.");
    }
    return { stopPolling: true, hasReceivedFirstToken };
  }
}
