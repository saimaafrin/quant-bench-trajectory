-- QuantBench Live: Supabase schema. Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to re-run.

-- Who may use the site. Add one row per approved email; add '*' to let anyone who signs up in.
create table if not exists public.allowed_emails (
  email text primary key
);
alter table public.allowed_emails enable row level security;   -- no policies: not readable or writable from the browser

insert into public.allowed_emails (email) values ('safrin@wm.edu') on conflict do nothing;

create or replace function public.is_allowed() returns boolean
language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.allowed_emails
    where email = '*' or lower(email) = lower(coalesce(auth.jwt() ->> 'email', ''))
  );
$$;
grant execute on function public.is_allowed() to authenticated;

-- One row per generated problem (a batch is several rows sharing batch_id).
create table if not exists public.runs (
  id         bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  user_id    uuid not null default auth.uid() references auth.users (id) on delete cascade,
  user_email text not null default (auth.jwt() ->> 'email'),
  batch_id   text,
  source     text,
  task       text,
  summary    jsonb not null,   -- compact metrics used by the history table and run reports
  record     jsonb not null    -- full outputs, tests, static-analysis findings
);
create index if not exists runs_batch_id on public.runs (batch_id);
create index if not exists runs_user_id on public.runs (user_id);

alter table public.runs enable row level security;

drop policy if exists "approved users read all runs" on public.runs;
create policy "approved users read all runs" on public.runs
  for select to authenticated using (public.is_allowed());

drop policy if exists "approved users add their own runs" on public.runs;
create policy "approved users add their own runs" on public.runs
  for insert to authenticated with check (public.is_allowed() and user_id = auth.uid());

drop policy if exists "users delete their own runs" on public.runs;
create policy "users delete their own runs" on public.runs
  for delete to authenticated using (user_id = auth.uid());

-- Managing access later (run in the SQL editor):
--   insert into public.allowed_emails values ('colleague@wm.edu');   -- approve one person
--   insert into public.allowed_emails values ('*');                  -- open to everyone who signs up
--   delete from public.allowed_emails where email = 'colleague@wm.edu';

-- Agent page (tool-calling trajectories): same rules as runs. Added 2026-10-28; safe to re-run.
create table if not exists public.agent_runs (
  id         bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  user_id    uuid not null default auth.uid() references auth.users (id) on delete cascade,
  user_email text not null default (auth.jwt() ->> 'email'),
  batch_id   text,
  task       text,
  summary    jsonb not null,   -- per-agent metrics for the history table and run reports
  record     jsonb not null    -- full trajectories (every step, call, result) and grading
);
create index if not exists agent_runs_batch_id on public.agent_runs (batch_id);
create index if not exists agent_runs_user_id on public.agent_runs (user_id);

alter table public.agent_runs enable row level security;

drop policy if exists "approved users read all agent runs" on public.agent_runs;
create policy "approved users read all agent runs" on public.agent_runs
  for select to authenticated using (public.is_allowed());

drop policy if exists "approved users add their own agent runs" on public.agent_runs;
create policy "approved users add their own agent runs" on public.agent_runs
  for insert to authenticated with check (public.is_allowed() and user_id = auth.uid());

drop policy if exists "users delete their own agent runs" on public.agent_runs;
create policy "users delete their own agent runs" on public.agent_runs
  for delete to authenticated using (user_id = auth.uid());
