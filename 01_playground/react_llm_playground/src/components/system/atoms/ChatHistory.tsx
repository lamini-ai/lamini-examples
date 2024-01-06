import React, { useCallback, useEffect, useRef } from "react";
import pinkLlama from "../../../assets/icon_lamini.png";
import userLogo from "../../../assets/icon_user.png";
import Markdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";

export interface ChatHistoryItem {
  entity: string;
  message: string;
}

interface ChatHistoryProps {
  chatHistory: Array<ChatHistoryItem>;
  isGenerating: boolean;
}

const CodeBlock = ({ language, value }) => {
  const normalizedLanguage = language
    ? language.replace("language-", "")
    : "python";
  return (
    <SyntaxHighlighter style={atomDark} language={normalizedLanguage}>
      {value}
    </SyntaxHighlighter>
  );
};

export default CodeBlock;

export const ChatHistory = (props: ChatHistoryProps) => {
  const scrollRef = useRef<HTMLInputElement>(null);
  const elemRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (elemRef.current === null || scrollRef.current === null) {
      return;
    }
    const bottom =
      scrollRef.current.scrollHeight -
      scrollRef.current.scrollTop -
      scrollRef.current.clientHeight;

    if (props.isGenerating && bottom < 100) {
      elemRef.current.scrollIntoView({ behavior: "auto", block: "end" });
    }
  }, [props.isGenerating, scrollRef.current?.scrollHeight]);

  return (
    <div className="w-full h-full rounded-[16px] overflow-hidden shadow-blackA7 bg-[#1d1d1d] border-[1px] border-[#393939] text-[#fff] font-sans text-[14px] p-4">
      <div
        ref={scrollRef}
        className="w-full h-full max-h-full rounded overflow-y-auto "
      >
        <div ref={elemRef} id="data" className="flex flex-col">
          {props.chatHistory.map((chat, index) => (
            <div className="flex w-full pb-2 first:mb-0 font-sans" key={index}>
              {chat.entity === "system" ? (
                chat.message === "pending" ? (
                  <div className="flex w-full">
                    <img
                      className="w-6 h-6 rounded-full shrink-0 mr-2"
                      src={pinkLlama}
                    ></img>
                    <div className="my-auto text-[4px] text-[#FF6C71] align-middle">
                      <div className="loading" />
                    </div>
                  </div>
                ) : (
                  <div className="mx-auto text-[#9E9E9E] text-[12px] whitespace-pre-wrap break-words">
                    <Markdown>{chat.message}</Markdown>
                  </div>
                )
              ) : (
                <div className="flex w-full">
                  {chat.entity === "user" ? (
                    <img
                      className="w-6 h-6 rounded-full shrink-0 mr-2"
                      src={userLogo}
                    ></img>
                  ) : (
                    <img
                      className="w-6 h-6 rounded-full shrink-0 mr-2"
                      src={pinkLlama}
                    ></img>
                  )}
                  <div className="my-auto text-[#9E9E9E] text-[12px] whitespace-pre-wrap break-words">
                    <Markdown
                      components={{
                        code: ({
                          node,
                          inline,
                          className,
                          children,
                          ...props
                        }) => {
                          const match = /language-(\w+)/.exec(className || "");
                          return (
                            <CodeBlock
                              language={match ? match[1] : null}
                              value={String(children).replace(/\n$/, "")}
                            />
                          );
                        },
                      }}
                    >
                      {chat.message}
                    </Markdown>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
