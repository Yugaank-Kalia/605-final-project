import Link from 'next/link';

export default function Home() {
	return (
		<div className='grid grid-rows-[auto_1fr_auto] min-h-screen p-8 pb-20 gap-16 sm:p-20'>
			{/* Hero Section */}
			<header className='flex flex-col items-center text-center gap-8 max-w-4xl mx-auto'>
				<h1 className='text-4xl sm:text-6xl font-bold tracking-tight'>
					Discover Your Perfect
				</h1>
				<span className='text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-500 text-4xl sm:text-6xl font-bold tracking-tight'>
					Music Match
				</span>
				<p className='text-xl text-foreground/70 max-w-2xl'>
					Experience personalized music recommendations powered by AI.
					Find new artists, tracks, and genres that perfectly match
					your taste.
				</p>
				<div className='flex gap-4 items-center flex-col sm:flex-row'>
					<Link
						className='rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] font-medium text-base h-12 px-8'
						href='/discover'
					>
						Start Discovering Free
					</Link>
				</div>
			</header>

			{/* Main Content */}
			<main className='flex flex-col gap-32'>
				{/* Features Grid */}
				<section className='grid sm:grid-cols-2 lg:grid-cols-3 gap-8'>
					{[
						{
							title: 'Smart Recommendations',
							description:
								"Our AI analyzes your music taste and listening habits to suggest tracks you'll love",
						},
						{
							title: 'Mood Detection',
							description:
								'Get playlist recommendations based on your current mood and activity',
						},
						{
							title: 'Genre Explorer',
							description:
								'Discover new genres and subgenres that match your musical preferences',
						},
						{
							title: 'Artist Discovery',
							description:
								'Find emerging artists similar to your favorites before they hit mainstream',
						},
						{
							title: 'Weekly Mixtapes',
							description:
								'Receive personalized weekly playlists curated just for you',
						},
						{
							title: 'Cross-Platform Sync',
							description:
								'Sync your listening history from Spotify, Apple Music, and other services',
						},
					].map((feature, i) => (
						<div
							key={i}
							className='p-6 rounded-2xl border border-black/[.08] dark:border-white/[.145] hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] transition-colors'
						>
							<h3 className='text-xl font-semibold mb-2'>
								{feature.title}
							</h3>
							<p className='text-foreground/70'>
								{feature.description}
							</p>
						</div>
					))}
				</section>
			</main>
		</div>
	);
}
