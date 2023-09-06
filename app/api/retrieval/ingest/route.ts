import { Document, Prisma } from "@prisma/client";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PrismaVectorStore } from "langchain/vectorstores/prisma";
import { NextRequest, NextResponse } from "next/server";
import { db } from "../../../../libs/db";

export const runtime = "nodejs";

// Before running, follow set-up instructions at
// https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase

/**
 * This handler takes input text, splits it into chunks, and embeds those chunks
 * into a vector store for later retrieval. See the following docs for more information:
 *
 * https://js.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
 * https://js.langchain.com/docs/modules/data_connection/vectorstores/integrations/supabase
 */
export async function POST(req: NextRequest) {
  const body = await req.json();
  const text = body.text;

  if (process.env.NEXT_PUBLIC_DEMO === "true") {
    return NextResponse.json(
      {
        error: [
          "Ingest is not supported in demo mode.",
          "Please set up your own version of the repo here: https://github.com/langchain-ai/langchain-nextjs-template",
        ].join("\n"),
      },
      { status: 403 },
    );
  }

  try {
    const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
      chunkSize: 256,
      chunkOverlap: 20,
    });

    const splitDocuments = await splitter.createDocuments([text]);

    // Use the `withModel` method to get proper type hints for `metadata` field:
    const vectorStore = PrismaVectorStore.withModel<Document>(db).create(
      new OpenAIEmbeddings(),
      {
        prisma: Prisma,
        tableName: "Document",
        vectorColumnName: "vector",
        columns: {
          id: PrismaVectorStore.IdColumn,
          content: PrismaVectorStore.ContentColumn,
        },
      },
    );

    await vectorStore.addModels(
      await db.$transaction(
        splitDocuments.map((content) =>
          db.document.create({ data: { content: content.pageContent } }),
        ),
      ),
    );
    // const vectorStore = await Chroma.fromDocuments(
    //   splitDocuments,
    //   new OpenAIEmbeddings(),
    //   {
    //     collectionName: "a-test-collection",
    //     url: "http://127.0.0.1:7080", // Optional, will default to this value
    //     collectionMetadata: {
    //       "hnsw:space": "cosine",
    //     }, // Optional, can be used to specify the distance method of the embedding space https://docs.trychroma.com/usage-guide#changing-the-distance-function
    //   },
    // );

    // await OpenSearchVectorStore.fromDocuments(
    //   splitDocuments,
    //   new OpenAIEmbeddings(),
    //   {
    //     client,
    //     indexName: "lookbook", // Will default to `documents`,
    //   },
    // );

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (e: any) {
    console.log({ e });
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
