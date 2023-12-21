import React from "react";
import { ChatHistory, ChatHistoryItem } from "../system/atoms/ChatHistory";
import { SendMessage } from "../system/atoms/SendMessage";
import fetchStreamingInference from "../../utils/fetchInference";

interface InteractivePlaygroundProps {
  token: string;
  default: string | null; // If there is a default model set, there will be no option to change the model
  custom_name: string;
  prompt_key?: string;
  shouldFetchModel: boolean;
  output_len: number;
}

export default function InteractivePlayground(
  props: InteractivePlaygroundProps
) {
  const [userTurn, setUserTurn] = React.useState<boolean>(true);
  const [message, setMessage] = React.useState("");

  const [chatHistory, setChatHistory] = React.useState<Array<ChatHistoryItem>>([
    { message: "What would you like to know?", entity: "model" },
  ]);

  const sendMessage = () => {
    setUserTurn(false);
    setChatHistory([
      ...chatHistory,
      { entity: "user", message },
      { entity: "system", message: "pending" },
    ]);
    const setNextModelResponse = (response: string) => {
      setChatHistory([
        ...chatHistory,
        { entity: "user", message },
        { entity: "model", message: response },
      ]);
    };
    fetchStreamingInference(
      message,
      props.default,
      setNextModelResponse,
      props.token,
      props.output_len
    ).then(() => {
      setUserTurn(true);
    });
  };

  return (
    <>
      <div className="grow mb-2 min-h-0">
        <ChatHistory
          chatHistory={chatHistory}
          isGenerating={!userTurn}
        ></ChatHistory>
      </div>
      <SendMessage
        message={message}
        onSendMessage={sendMessage}
        setMessage={setMessage}
        userTurn={userTurn}
      ></SendMessage>
    </>
  );
}
