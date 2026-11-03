-- Supabase only: the private bucket the upload page writes to (through signed upload URLs).
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('cortex-docs', 'cortex-docs', false, 52428800,
        array['application/pdf',
              'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
              'text/plain'])
on conflict (id) do nothing;
