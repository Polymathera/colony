/**
 * Markdown renderer for KB chunks — handles the multimodal
 * extensions the layout-aware readers (Mistral OCR, Anthropic
 * native PDF, Marker / Docling / MinerU) emit:
 *
 * - ``colony-image://<sha>`` image URIs are rewritten to
 *   ``/api/v1/kb/images/<sha>`` so a browser ``<img>`` tag resolves
 *   them through the dashboard's auth-gated figure endpoint.
 * - GFM tables, fenced code blocks, ``$...$`` math survive the
 *   ``MarkdownChunker``'s atomic-block discipline; rendering them
 *   here is straight ``react-markdown`` + ``remark-gfm``.
 * - Long blocks are wrapped in a max-height container with a fade
 *   so a single oversized chunk does not blow up the chat panel
 *   layout. Operators expand on demand.
 *
 * Used by both the source-drilldown chunk list and the search-hit
 * preview list so a chunk's visual rendering is consistent
 * everywhere it appears.
 */
import { useState } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

const COLONY_IMAGE_PREFIX = "colony-image://";

/**
 * Rewrite a ``colony-image://<sha>`` URI to its dashboard-resolved
 * HTTP form. Other schemes (https://, /relative, mailto:) pass
 * through unchanged.
 */
function resolveImageUrl(src: string | undefined): string | undefined {
  if (!src) return src;
  if (src.startsWith(COLONY_IMAGE_PREFIX)) {
    const sha = src.slice(COLONY_IMAGE_PREFIX.length);
    return `/api/v1/kb/images/${encodeURIComponent(sha)}`;
  }
  return src;
}

/** Markdown ``<img>`` override that rewrites colony-image:// URIs. */
function ChunkImage(
  props: React.ImgHTMLAttributes<HTMLImageElement>,
) {
  const { src, alt, ...rest } = props;
  const resolved = resolveImageUrl(typeof src === "string" ? src : undefined);
  return (
    <img
      {...rest}
      src={resolved}
      alt={alt ?? ""}
      className={cn(
        "my-2 max-w-full rounded border border-border bg-muted/20",
        "max-h-[24rem] object-contain",
        rest.className,
      )}
      loading="lazy"
    />
  );
}

/**
 * Wrap fenced code / pre blocks in a max-height container. KB chunks
 * may carry a whole page of dense source code; without this they
 * scroll the entire chunk list out of the viewport.
 */
function ChunkPre(
  props: React.HTMLAttributes<HTMLPreElement> & { children?: React.ReactNode },
) {
  const { children, className, ...rest } = props;
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="not-prose my-2 rounded-md border border-border bg-muted/30 overflow-hidden">
      <pre
        {...rest}
        className={cn(
          "m-0 px-3 py-2 overflow-auto text-[10px] leading-snug whitespace-pre-wrap break-words",
          !expanded && "max-h-[12rem] [mask-image:linear-gradient(to_bottom,black_70%,transparent)]",
          className,
        )}
      >
        {children}
      </pre>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full border-t border-border bg-muted/40 px-3 py-1 text-[10px] text-muted-foreground hover:bg-muted/60 transition-colors text-left"
      >
        <span className="mr-1.5">{expanded ? "▼" : "▶"}</span>
        {expanded ? "Collapse" : "Expand"}
      </button>
    </div>
  );
}

const COMPONENTS: Components = {
  img: ChunkImage,
  pre: ChunkPre,
};

export function ChunkMarkdown({ text }: { text: string }) {
  return (
    <div
      className={cn(
        "prose prose-invert prose-xs max-w-none leading-5",
        "[&_table]:text-[10px] [&_th]:px-2 [&_td]:px-2",
        "[&_code]:text-[10px] [&_pre]:text-[10px]",
        "[&_h1]:text-sm [&_h2]:text-xs [&_h3]:text-xs",
      )}
    >
      <Markdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {text}
      </Markdown>
    </div>
  );
}
