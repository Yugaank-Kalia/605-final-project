'use client';

import Image from 'next/image';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import {
	Form,
	FormField,
	FormItem,
	FormControl,
	FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface Message {
	id: string;
	type: 'user' | 'assistant';
	content: string;
	recommendations?: {
		url: string;
		title: string;
		artist: string;
		imageUrl: string;
		genre: string[];
		release_date: string;
	}[];
	requestedSong?: {
		url: string;
		title: string;
		artist: string;
		imageUrl: string;
		genre: string[];
		release_date: string;
	};
}

const initialMessage = {
	id: 'typing',
	type: 'assistant' as const,
	content: '',
};

const formSchema = z.object({
	spotify_url: z.string().url('Please provide a valid Spotify track URL'),
});

const getRecommendations = async (
	spotify_url: string
): Promise<{
	recommendations: Message['recommendations'];
	requestedSong: Message['requestedSong'];
}> => {
	const response = await fetch(
		'https://00db-3-138-102-164.ngrok-free.app/api/recommend', // change before class
		{
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ spotify_url }),
		}
	);

	if (!response.ok) throw new Error('Failed to fetch recommendations');
	const data = await response.json();

	return {
		recommendations: data.recommendations,
		requestedSong: {
			url: data.input_track_url,
			title: data.input_track_title,
			artist: data.input_track_artist,
			imageUrl: data.input_track_imageUrl,
			genre: data.input_track_genre,
			release_date: data.input_track_release_date,
		},
	};
};

export default function DiscoverPage() {
	const [messages, setMessages] = useState<Message[]>([initialMessage]);
	const [isLoading, setIsLoading] = useState(false);
	const messagesEndRef = useRef<HTMLDivElement>(null);

	const form = useForm({
		resolver: zodResolver(formSchema),
		defaultValues: { spotify_url: '' },
	});

	useEffect(() => {
		setTimeout(() => {
			setMessages([
				{
					id: '1',
					type: 'assistant',
					content:
						'Hi! I can help you discover new music, paste a Spotify link to a song you like.',
				},
			]);
		}, 1000);
	}, []);

	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	};

	useEffect(() => {
		scrollToBottom();
	}, [messages]);

	const onSubmit = async (values: { spotify_url: string }) => {
		if (isLoading) return;

		const userMessage: Message = {
			id: Date.now().toString(),
			type: 'user',
			content: values.spotify_url,
		};

		setMessages((prev) => [...prev, userMessage]);
		setIsLoading(true);
		setMessages((prev) => [
			...prev,
			{ id: 'typing', type: 'assistant', content: '' },
		]);

		try {
			const { requestedSong, recommendations } = await getRecommendations(
				values.spotify_url
			);
			setMessages((prev) => prev.filter((msg) => msg.id !== 'typing'));

			const aiResponse: Message = {
				id: (Date.now() + 1).toString(),
				type: 'assistant',
				content:
					'Based on your interests, here are some recommendations:',
				requestedSong: requestedSong,
				recommendations: recommendations,
			};
			setMessages((prev) => [...prev, aiResponse]);
			form.reset();
		} catch (error) {
			setMessages((prev) => prev.filter((msg) => msg.id !== 'typing'));
			setMessages((prev) => [
				...prev,
				{
					id: (Date.now() + 1).toString(),
					type: 'assistant',
					content: `Sorry, I encountered an error, ${error} while getting recommendations. Please check the Spotify URL.`,
				},
			]);
		} finally {
			setIsLoading(false);
		}
	};

	const handleClearChat = () => {
		setMessages([initialMessage]);
		form.reset();
		setIsLoading(false);

		setTimeout(() => {
			setMessages([
				{
					id: '1',
					type: 'assistant',
					content:
						'Hi! I can help you discover new music, paste a Spotify link to a song you like.',
				},
			]);
		}, 1000);
	};

	return (
		<div className='flex flex-col h-screen max-h-screen'>
			<header className='flex items-center justify-between p-4 border-b border-black/10 dark:border-white/10'>
				<h1 className='text-xl font-semibold'>Music Discovery</h1>
				<button
					onClick={handleClearChat}
					className='rounded-full bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 text-sm font-medium hover:opacity-90'
				>
					Clear Chat
				</button>
			</header>

			<div className='flex-1 overflow-y-auto p-4 space-y-4'>
				{/* Chat + Recommendations */}
				{messages.map((message) => (
					<div
						key={message.id}
						className={`flex ${
							message.type === 'user'
								? 'justify-end'
								: 'justify-start'
						}`}
					>
						<div
							className={`max-w-[80%] rounded-md p-4 ${
								message.type === 'user'
									? 'bg-blue-500 text-white'
									: 'bg-zinc-100 dark:bg-zinc-800'
							}`}
						>
							{message.id === 'typing' ? (
								<div className='flex gap-2 h-6 items-center'>
									<div className='w-2 h-2 rounded-full bg-blue-500 animate-pulse'></div>
									<div className='w-2 h-2 rounded-full bg-blue-500 animate-pulse delay-150'></div>
									<div className='w-2 h-2 rounded-full bg-blue-500 animate-pulse delay-300'></div>
								</div>
							) : (
								<div className='flex flex-col gap-4'>
									<p>{message.content}</p>
									{message.recommendations && (
										<>
											{message.requestedSong && (
												<div className='max-w-xs mb-4'>
													<p className='text-sm font-medium text-zinc-500 dark:text-zinc-400 mb-2'>
														Requested Song
													</p>
													<div className='flex flex-col items-center bg-white dark:bg-black rounded-lg p-4 shadow-md'>
														<div className='relative w-full aspect-square'>
															<Image
																src={
																	message
																		.requestedSong
																		.imageUrl
																}
																alt={
																	message
																		.requestedSong
																		.title
																}
																className='rounded-md object-cover'
																fill
															/>
														</div>
														<div className='mt-3 w-full space-y-1'>
															<h3 className='font-semibold text-sm truncate'>
																{
																	message
																		.requestedSong
																		.title
																}
															</h3>
															<p className='text-sm text-zinc-600 dark:text-zinc-400 truncate'>
																{
																	message
																		.requestedSong
																		.artist
																}
															</p>
															<div className='flex gap-2 mt-1 flex-wrap'>
																<Badge variant='secondary'>
																	{message
																		.requestedSong
																		.genre &&
																	message
																		.requestedSong
																		.genre
																		.length >
																		0
																		? message
																				.requestedSong
																				.genre[0]
																		: 'no genre'}
																</Badge>
																<Badge variant='outline'>
																	{
																		message
																			.requestedSong
																			.release_date
																	}
																</Badge>
															</div>
														</div>
													</div>
												</div>
											)}
											{/* Recommendations Grid */}
											<div className=''>
												<p className='text-sm font-medium text-zinc-500 dark:text-zinc-400 mb-2'>
													Recommendations
												</p>
												<div className='grid grid-cols-1 sm:grid-cols-5 gap-4 mt-4'>
													{message.recommendations.map(
														(rec, index) => (
															<Link
																key={index}
																href={rec.url}
																target='_blank'
																className='w-full'
															>
																<div className='max-w-xs mb-4'>
																	<div className='flex flex-col items-center bg-white dark:bg-black rounded-lg p-4 shadow-md'>
																		<div className='relative w-full aspect-square'>
																			<Image
																				src={
																					rec.imageUrl
																				}
																				alt={
																					rec.title
																				}
																				className='rounded-md object-cover'
																				fill
																			/>
																		</div>
																		<div className='mt-3 w-full space-y-1'>
																			<h3 className='font-semibold text-sm truncate'>
																				{
																					rec.title
																				}
																			</h3>
																			<p className='text-sm text-zinc-600 dark:text-zinc-400 truncate'>
																				{
																					rec.artist
																				}
																			</p>
																			<div className='flex gap-2 mt-1 flex-wrap'>
																				<Badge variant='secondary'>
																					{rec.genre &&
																					rec
																						.genre
																						.length >
																						0
																						? rec
																								.genre[0]
																						: 'no genre'}
																				</Badge>
																				<Badge variant='outline'>
																					{
																						rec.release_date
																					}
																				</Badge>
																			</div>
																		</div>
																	</div>
																</div>
															</Link>
														)
													)}
												</div>
											</div>
										</>
									)}
								</div>
							)}
						</div>
					</div>
				))}
				<div ref={messagesEndRef} />
			</div>

			<div className='border-t border-black/10 dark:border-white/10 p-4'>
				<Form {...form}>
					<form
						onSubmit={form.handleSubmit(onSubmit)}
						className='flex gap-4 max-w-4xl mx-auto'
					>
						<FormField
							control={form.control}
							name='spotify_url'
							render={({ field }) => (
								<FormItem className='flex-1'>
									<FormControl>
										<Input
											{...field}
											placeholder='Paste a Spotify track URL...'
											disabled={isLoading}
											className='rounded-full px-4 py-2'
										/>
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
						<Button
							type='submit'
							disabled={isLoading}
							className='rounded-full px-6 py-2'
						>
							{isLoading ? 'Thinking...' : 'Send'}
						</Button>
					</form>
				</Form>
			</div>
		</div>
	);
}
