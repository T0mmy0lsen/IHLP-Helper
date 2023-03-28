import {Main, Button, Action, Typography, Route, Sider, Section, Get} from 'react-antd-admin-panel';
import {faSearch, faTable} from "@fortawesome/free-solid-svg-icons";
import Error from "./views/Error";
import Search from "./views/Search";
import Schedule from "./views/Schedule";

let isProd = process.env.NODE_ENV === 'production';

export default {
    config: {
        debug: isProd ? 0 : 1,
        drawer: { collapsed: false },
        pathToApi: isProd ? 'https://dispatchassistant.sdu.dk/ihlp' : 'http://127.0.0.1:8000/ihlp',
        pathToLogo: undefined, // { src: '/logo192.png', height: 30, width: 30 },
        fallbackApi: 'http://localhost',
        fallbackApiOn: [404],
        defaultRoute: () => '/',
        profile: (next: any) => {
            next(new Section()
                .addRowEnd([
                    new Typography()
                        .style({ marginTop: 1 })
                        .label(`IHLP Dispatch Assistant`)
                ])
            );
        },
        access: {
            accessViolationRoute: (main: Main) => {
                main.$route(`/error`)
            },
            accessViolationApi: (main: Main) => {
                main.tsxErrorMessage('Du har ikke adgang til denne funktionalitet');
            },
            access: () =>
            {
                return ({ hidden: false, access: true });
            },
        },
        bootFailed: () => {
            window.location.href = '/';
        },
        boot: (main: Main, next: any) => {
            next()
        }
    },
    routes: [
        new Route().key('/')
            .exact()
            .component(Search),
        new Route().key('/schedule')
            .exact()
            .component(Schedule),
        new Route().key('/error')
            .exact()
            .component(Error),
    ],
    drawer: (next: any) => {
        next(new Section().add(new Sider()
            .add(new Button().access(true)
                .action(new Action().route(() => '/').label('Search').fontawesome(faSearch)))
            .add(new Button().access(true)
                .action(new Action().route(() => '/schedule').label('Schedule').fontawesome(faTable)))
        ));
    },
}